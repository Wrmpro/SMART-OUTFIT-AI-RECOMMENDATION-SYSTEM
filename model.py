import cv2
import numpy as np
import pandas as pd
from PIL import ImageColor

SKIN_TONES = {
    "#373028": "Deepest Skin",
    "#422811": "Very Deep",
    "#513B2E": "Deep Brown",
    "#6F503C": "Medium Brown",
    "#81654F": "Tan",
    "#9D7A54": "Light Tan",
    "#BEA07E": "Medium Fair",
    "#E5C8A6": "Light Fair",
    "#E7C1B8": "Warm Fair",
    "#F3DAD6": "Very Fair",
    "#FBF2F3": "Pale"
}

SKIN_RGB_MAP = {k: np.array(ImageColor.getrgb(k)) for k in SKIN_TONES}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_skin_tone_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return "Medium Fair"

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    h, w, _ = face_rgb.shape
    center = face_rgb[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
    avg_rgb = center.mean(axis=(0,1))

    closest_hex = min(
        SKIN_RGB_MAP,
        key=lambda k: np.linalg.norm(avg_rgb - SKIN_RGB_MAP[k])
    )

    return SKIN_TONES[closest_hex]

styles = pd.read_csv("data/styles.csv", on_bad_lines="skip")
images = pd.read_csv("data/images.csv")

OCCASION_RULES = {
    "Casual": ["Casual"],
    "Party": ["Party", "Evening"],
    "Formal": ["Formal", "Office"],
    "Ethnic": ["Ethnic", "Festive"]
}

WEATHER_RULES = {
    "Hot": ["Summer"],
    "Mild": ["Spring", "Autumn"],
    "Cold": ["Winter"],
    "Rainy": ["Monsoon"]
}

def get_image(pid):
    row = images[images["filename"] == f"{int(pid)}.jpg"]
    return None if row.empty else row.iloc[0]["link"]

def recommend_outfits(skin_tone, gender, occasion, weather, n=5):
    df = styles.copy()
    df = df[df["gender"].str.lower() == gender.lower()]
    df = df[df["usage"].isin(OCCASION_RULES.get(occasion, []))]
    df = df[df["season"].isin(WEATHER_RULES.get(weather, []))]

    tops = df[df["articleType"].isin(["Tshirts","Shirts","Tops","Kurtas"])]
    bottoms = df[df["articleType"].isin(["Jeans","Trousers","Skirts","Palazzos"])]

    outfits = []

    for _ in range(n):
        t = tops.sample(1).iloc[0]
        b = bottoms.sample(1).iloc[0]

        outfits.append({
            "top_name": t["productDisplayName"],
            "top_image_link": get_image(t["id"]),
            "bottom_name": b["productDisplayName"],
            "bottom_image_link": get_image(b["id"]),
            "confidence": 100.0
        })

    return outfits
