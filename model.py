import cv2
import numpy as np
import pandas as pd
from PIL import ImageColor
import os
from groq import Groq
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq Client safely
api_key = os.environ.get("GROQ_API_KEY")
client = None
if api_key:
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Groq client: {e}")

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

# Color map for wardrobe recommendations
SKINTONE_TOP_COLORS = {
    "Deep Brown": ["Red", "Turquoise", "Pink", "Blue", "Copper", "Green", "Sea Green", "Brown", "Orange", "Peach"],
    "Deepest Skin": ["Pink", "Purple", "Turquoise", "Blue", "Copper", "Gold", "Lime", "Navy", "Orange", "Brown"],
    "Light Fair": ["Lavender", "Sea Green", "Tan", "Beige", "Turquoise", "Peach", "Gold"],
    "Light Tan": ["Lavender", "Tan", "Sea Green", "Gold", "Lime", "Peach", "Beige", "Turquoise", "Yellow"],
    "Medium Brown": ["Purple", "Red", "Teal", "Turquoise", "Green", "Pink", "Blue", "Brown", "Gold"],
    "Medium Fair": ["Tan", "Lavender", "Sea Green", "Peach", "Turquoise", "Gold", "Beige"],
    "Pale": ["Tan", "Lavender", "Sea Green", "Peach", "Beige", "Turquoise"],
    "Tan": ["Lime", "Green", "Lavender", "Tan", "Gold", "Peach", "Turquoise", "Yellow"],
    "Very Deep": ["Blue", "Red", "Pink", "Purple", "Teal", "Gold", "Olive", "Orange", "Peach", "Turquoise"],
    "Very Fair": ["Tan", "Lavender", "Sea Green", "Peach", "Beige"],
    "Warm Fair": ["Sea Green", "Peach", "Tan", "Lavender", "Beige", "Turquoise", "Gold"]
}

# Expanded Party Outfit Data from Myntra and Bonsoir
import json
try:
    with open('party_outfits_expanded.json', 'r') as f:
        PARTY_OUTFITS_KB = json.load(f)
except Exception:
    PARTY_OUTFITS_KB = {"Male": [], "Female": []}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_skin_tone_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Medium Fair"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        return "Medium Fair"

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    avg_color = face_rgb.mean(axis=(0, 1))
    avg_hex = "#{:02x}{:02x}{:02x}".format(int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
    
    distances = {hex_code: np.linalg.norm(SKIN_RGB_MAP[hex_code] - avg_color) for hex_code in SKIN_RGB_MAP}
    closest_hex = min(distances, key=distances.get)
    
    return SKIN_TONES.get(closest_hex, "Medium Fair")

def get_skin_tone_category(skin_tone):
    """Map detected skin tone to category"""
    skin_tone_lower = skin_tone.lower()
    if any(x in skin_tone_lower for x in ["fair", "pale", "light"]):
        return "Light/Fair Skin"
    elif any(x in skin_tone_lower for x in ["medium", "olive", "tan", "warm"]):
        return "Medium/Olive Skin"
    else:
        return "Dark/Deep Skin"

# Load CSV files with error handling for malformed lines
try:
    # Use on_bad_lines='skip' to handle the ParserError reported by the user
    styles = pd.read_csv("styles.csv", on_bad_lines='skip')
except Exception as e:
    print(f"Error loading styles.csv: {e}")
    styles = pd.DataFrame()

try:
    images = pd.read_csv("images.csv", on_bad_lines='skip')
except Exception as e:
    print(f"Error loading images.csv: {e}")
    images = pd.DataFrame()

try:
    wardrobe = pd.read_csv("Wardrobe Assistant.csv", on_bad_lines='skip')
except Exception as e:
    print(f"Error loading Wardrobe Assistant.csv: {e}")
    wardrobe = pd.DataFrame()

OCCASION_RULES = {
    "Casual": ["Casual", "casual"],
    "Formal": ["Formal", "formal"],
    "Party": ["Party", "party"],
    "Festival": ["Festival", "festival"]
}

def get_image(pid):
    if images.empty:
        return None
    row = images[images["filename"] == f"{int(pid)}.jpg"]
    return None if row.empty else row.iloc[0]["link"]

def get_batch_ai_rationales(skin_tone, occasion, outfits_list):
    """Generate AI rationales for multiple outfits in a single call to save credits and reduce latency"""
    if not client or not outfits_list:
        return [f"This {occasion.lower()} outfit complements your {skin_tone} skin tone beautifully." for _ in outfits_list]

    outfits_text = "\n".join([f"{i+1}. {o}" for i, o in enumerate(outfits_list)])
    prompt = f"""Provide a unique 1-sentence fashion rationale for each of the following {len(outfits_list)} outfits.
Skin Tone: {skin_tone}
Occasion: {occasion}

Outfits:
{outfits_text}

Return exactly {len(outfits_list)} sentences, one per line, starting with the number. Keep them concise and stylish."""
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        response = completion.choices[0].message.content.strip().split('\n')
        # Extract the sentence after the number
        rationales = []
        for line in response:
            parts = line.split('. ', 1)
            if len(parts) > 1:
                rationales.append(parts[1].strip())
            else:
                rationales.append(line.strip())
        
        # Ensure we have the right number of rationales
        while len(rationales) < len(outfits_list):
            rationales.append(f"This {occasion.lower()} outfit is perfectly suited to your {skin_tone} complexion.")
        return rationales[:len(outfits_list)]
    except Exception as e:
        print(f"Groq API Error: {e}")
        return [f"This {occasion.lower()} outfit complements your {skin_tone} skin tone beautifully." for _ in outfits_list]

def recommend_party_outfits(skin_tone, gender, n=5):
    """Recommend party outfits using real product data from Myntra and Bonsoir"""
    gender_key = "Male" if gender.lower() == "men" else "Female"
    
    if gender_key not in PARTY_OUTFITS_KB or not PARTY_OUTFITS_KB[gender_key]:
        return []
    
    kb_outfits = PARTY_OUTFITS_KB[gender_key]
    sample_size = min(n, len(kb_outfits))
    sampled = random.sample(kb_outfits, sample_size)
    
    # Extract descriptions and call Groq API once for all rationales
    outfit_descriptions = [o['name'] for o in sampled]
    rationales = get_batch_ai_rationales(skin_tone, "Party", outfit_descriptions)
    
    outfits = []
    for i, outfit in enumerate(sampled):
        image_url = outfit['image']
        # Ensure HTTPS
        if image_url.startswith("http://"):
            image_url = image_url.replace("http://", "https://")
            
        outfits.append({
            "top_name": outfit['name'],
            "top_image_link": image_url,
            "bottom_name": "Matching Piece",
            "bottom_image_link": image_url,
            "rationale": rationales[i],
            "is_fallback": False,
            "color": outfit.get('color', ''),
            "is_party": False  # Set to False to use the same layout as Festival for better image display
        })
    
    return outfits

def recommend_festival_outfits(skin_tone, gender, n=5):
    """Recommend ONLY from Wardrobe Assistant CSV for Festival (STRICT)"""
    if wardrobe.empty:
        return []
    
    df = wardrobe.copy()
    
    # Filter by gender
    gender_keywords = ["women", "female", "girls"] if gender.lower() in ["women", "female"] else ["men", "male", "boys"]
    gender_mask = False
    for kw in gender_keywords:
        gender_mask = (df["occasion"].str.contains(kw, case=False, na=False)) | gender_mask
    
    # Filter by festival keywords
    festival_keywords = ["festival", "festive", "celebration"]
    occ_mask = False
    for kw in festival_keywords:
        occ_mask = (df["occasion"].str.contains(kw, case=False, na=False)) | occ_mask
    
    df = df[occ_mask]
    
    # Skin tone color filter
    allowed_colors = SKINTONE_TOP_COLORS.get(skin_tone, [])
    if allowed_colors and "color" in df.columns:
        col_mask = False
        for c in allowed_colors:
            col_mask = df["color"].str.contains(c, case=False, na=False) | col_mask
        df = df[col_mask]
    
    # Must have image
    if "Image URL-src" in df.columns:
        df = df[df["Image URL-src"].notna()]
    
    if df.empty:
        # Fallback if no matching festival outfits found in CSV
        return []
    
    sampled = df.sample(min(n, len(df)))
    outfits = []
    
    outfit_descriptions = [row.get("product_name", "Festival Outfit") for _, row in sampled.iterrows()]
    rationales = get_batch_ai_rationales(skin_tone, "Festival", outfit_descriptions)
    
    for i, (_, row) in enumerate(sampled.iterrows()):
        image_url = row.get("Image URL-src", "")
        # Fix: Ensure HTTPS for images
        if image_url.startswith("http://"):
            image_url = image_url.replace("http://", "https://")
            
        outfits.append({
            "top_name": row.get("product_name", "Festival Wear"),
            "top_image_link": image_url,
            "bottom_name": "Matching Piece",
            "bottom_image_link": image_url,
            "rationale": rationales[i],
            "is_fallback": False,
            "is_party": False
        })
    
    return outfits

def recommend_outfits(skin_tone, gender, occasion, weather, n=5):
    """Main recommendation function"""
    occ_lower = occasion.lower()
    
    # Party: Use real product data
    if occ_lower == "party":
        return recommend_party_outfits(skin_tone, gender, n)
    
    # Festival: Use ONLY Wardrobe CSV (STRICT)
    if occ_lower == "festival":
        return recommend_festival_outfits(skin_tone, gender, n)
    
    # Casual & Formal: Use Myntra dataset
    if styles.empty:
        return []

    df = styles.copy()
    df = df[df["gender"].str.lower() == gender.lower()]
    
    myntra_occ = OCCASION_RULES.get(occasion, [occasion])
    df = df[df["usage"].isin(myntra_occ)]
    
    tops = df[df["articleType"].isin(["Tshirts", "Shirts", "Tops", "Kurtas", "Jackets"])]
    bottoms = df[df["articleType"].isin(["Jeans", "Trousers", "Skirts", "Palazzos", "Shorts"])]

    if tops.empty or bottoms.empty:
        return []

    outfits = []
    sample_size = min(n, len(tops), len(bottoms))
    
    # Randomly sample tops and bottoms to create unique combinations
    sampled_tops = tops.sample(sample_size)
    sampled_bottoms = bottoms.sample(sample_size)
    
    outfit_descriptions = [f"{sampled_tops.iloc[i]['productDisplayName']} with {sampled_bottoms.iloc[i]['productDisplayName']}" for i in range(sample_size)]
    rationales = get_batch_ai_rationales(skin_tone, occasion, outfit_descriptions)

    for i in range(sample_size):
        t = sampled_tops.iloc[i]
        b = sampled_bottoms.iloc[i]
        
        top_img = get_image(t["id"])
        bot_img = get_image(b["id"])
        
        # Ensure HTTPS
        if top_img and top_img.startswith("http://"):
            top_img = top_img.replace("http://", "https://")
        if bot_img and bot_img.startswith("http://"):
            bot_img = bot_img.replace("http://", "https://")

        outfits.append({
            "top_name": t["productDisplayName"],
            "top_image_link": top_img,
            "bottom_name": b["productDisplayName"],
            "bottom_image_link": bot_img,
            "rationale": rationales[i],
            "is_fallback": False,
            "is_party": False
        })

    return outfits
