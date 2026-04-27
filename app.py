from flask import Flask, request, render_template
import pandas as pd
import random
import os
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ================= PATH FIX =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

trending_products = pd.read_csv(os.path.join(BASE_DIR, "models/trending_products.csv"))
train_data = pd.read_csv(os.path.join(BASE_DIR, "models/clean_data.csv"))

# ================= DATABASE =================
app.secret_key = "secretkey123"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///ecommerce.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ================= MODELS =================
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    email = db.Column(db.String(100))
    password = db.Column(db.String(100))

class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))

# ================= FUNCTIONS =================
def truncate(text, length):
    return text[:length] + "..." if len(text) > length else text

def content_based_recommendations(data, item_name, top_n=10):
    if item_name not in data['Name'].values:
        return pd.DataFrame()

    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(data['Tags'])
    similarity = cosine_similarity(matrix, matrix)

    idx = data[data['Name'] == item_name].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    indices = [i[0] for i in scores]

    return data.iloc[indices][['Name','ReviewCount','Brand','ImageURL','Rating']]

# ================= STATIC =================
images = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]

# ================= ROUTES =================
@app.route("/")
def home():
    imgs = [random.choice(images) for _ in range(len(trending_products))]
    return render_template(
        "index.html",
        trending_products=trending_products.head(8),
        truncate=truncate,
        random_product_image_urls=imgs,
        random_price=random.randint(50, 200)
    )

@app.route("/main")
def main():
    return render_template("main.html", content_based_rec=pd.DataFrame())

@app.route("/recommendations", methods=["POST"])
def recommendations():
    product = request.form.get("prod")
    n = request.form.get("nbr")

    if not product or not n:
        return render_template(
            "main.html",
            message="Enter valid input",
            content_based_rec=pd.DataFrame()
        )

    result = content_based_recommendations(train_data, product, int(n))

    if result.empty:
        return render_template(
            "main.html",
            message="Product not found",
            content_based_rec=pd.DataFrame()
        )

    return render_template(
        "main.html",
        content_based_rec=result,
        truncate=truncate,
        random_price=random.randint(50, 200)
    )

# ================= INIT DB =================
with app.app_context():
    db.create_all()

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)