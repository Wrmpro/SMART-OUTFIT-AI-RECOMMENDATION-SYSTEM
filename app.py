from flask import Flask, render_template, request, jsonify
import os
from model import detect_skin_tone_from_image, recommend_outfits

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/customize")
def customize():
    return render_template("customize.html")

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    image = request.files["image"]
    gender = request.form.get("gender", "women")
    occasion = request.form.get("occasion", "Casual")
    weather = request.form.get("weather", "Hot")

    # Save image
    filename = image.filename
    path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(path)

    # 🔥 ADD THIS LINE (IMPORTANT)
    image_path = f"/static/uploads/{filename}"

    skin_tone = detect_skin_tone_from_image(path)

    outfits = recommend_outfits(
        skin_tone=skin_tone,
        gender=gender,
        occasion=occasion,
        weather=weather
    )

    return jsonify({
        "skin_tone": skin_tone,
        "outfits": outfits,
        "user_image": image_path   # 🔥 ADDED
    })

if __name__ == "__main__":
    app.run(debug=True)
