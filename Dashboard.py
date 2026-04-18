import streamlit as st
import cv2
import numpy as np
import zipfile
import io
from deepface import DeepFace

st.set_page_config(page_title="Ultimate Pro Sorter", layout="wide")

def get_category(img, file_bytes):
    try:
        # 1. THE HUMAN CHECK (Precision Focus)
        results = DeepFace.analyze(img, actions=['emotion'], detector_backend='retinaface', silent=True)
        e = results[0]['emotion']
        
        # MOOD SPECTRUM LOGIC
        if e['happy'] > 80: return "1_Happy/Ecstatic"
        if e['happy'] > 20: return "1_Happy/Pleasant_Smile"
        if e['neutral'] > 50: return "2_Neutral/Relaxed"
        if e['neutral'] > 20: return "2_Neutral/Serious_Candid"
        if e['sad'] > 30: return "3_Sad/Melancholy"
        return "2_Neutral/Mixed_Expression"

    except:
        # 2. THE "NOT HUMAN" CHECK (Dog vs Doc vs Scene)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Document Check: High edges + specific Aspect Ratio
        edges = cv2.countNonZero(cv2.Canny(gray, 100, 200))
        density = (edges / (img.shape[0] * img.shape[1])) * 100
        h, w = img.shape[:2]
        is_tall = h > w
        
        # A dog usually has a lot of texture but isn't 'structured' like a page
        if density > 4.5 and is_tall:
            return "4_Professional/Documents_Certificates"
        elif 1.5 < density < 4.0:
            # Most dog photos/animals fall in this medium-texture range
            return "5_Animals_and_Pets"
        else:
            return "6_Scenery_and_Backgrounds"

st.title("📸 Ultimate AI Photographer Assistant")
files = st.file_uploader("Upload Batch", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if files and st.button("🚀 Run Deep Sort"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a") as zip_f:
        pbar = st.progress(0)
        for i, file in enumerate(files):
            # Read image for analysis
            img_data = file.read()
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            category = get_category(img, img_data)
            zip_f.writestr(f"{category}/{file.name}", img_data)
            pbar.progress((i + 1) / len(files))
            
    st.download_button("📂 Download 100% Sorted ZIP", zip_buffer.getvalue(), "AI_Final_Sort.zip")