from flask import Flask, render_template, request, jsonify
import os
import time
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
    image = request.files.get("image")
    if not image:
        return jsonify({"error": "No image uploaded"}), 400
        
    gender = request.form.get("gender", "women")
    occasion = request.form.get("occasion", "Casual")
    weather = request.form.get("weather", "Hot")

    # Save image with timestamp to prevent caching issues
    timestamp = int(time.time())
    filename = f"{timestamp}_{image.filename}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(path)

    # Relative path for frontend
    image_path = f"/static/uploads/{filename}"

    # Analyze
    skin_tone = detect_skin_tone_from_image(path)

    # Recommend
    outfits = recommend_outfits(
        skin_tone=skin_tone,
        gender=gender,
        occasion=occasion,
        weather=weather
    )

    return jsonify({
        "skin_tone": skin_tone,
        "outfits": outfits,
        "user_image": image_path
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
