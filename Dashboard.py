import streamlit as st
import cv2
import numpy as np
import zipfile
import io
from deepface import DeepFace

st.set_page_config(page_title="Pro-Cull Nuance", layout="wide")

def get_pro_category(img):
    # 1. TECHNICAL BLUR CHECK (Standard for pros)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 100:
        return "4_Technical_Reject/Blurry_Photos"

    try:
        # Normalize lighting to force geometric recognition over 'lighting vibes'
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        norm_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        results = DeepFace.analyze(norm_img, actions=['emotion'], detector_backend='retinaface', silent=True)
        e = results[0]['emotion']
        
        # --- NEW NUANCED CATEGORIES ---
        
        # A. THE SMILE SPECTRUM
        if e['happy'] > 95: 
            return "1_Selections/High_Energy_Laugh" # Vibrant, open mouth
        if e['happy'] > 40: 
            return "1_Selections/Subtle_Social_Smile" # Posed/Polite smile
            
        # B. THE "PORTRAIT" SPECTRUM (Serious/Neutral)
        if e['neutral'] > 80:
            return "1_Selections/Neutral_High_Focus" # Serious editorial look
        if e['neutral'] > 30:
            return "2_Candids/Relaxed_Natural" # Softer, non-smiling but calm
            
        # C. STORY & EMOTION (The "Artistic" shots)
        if e['surprise'] > 40:
            return "2_Candids/Surprise_Candid" # Great for wedding toasts/reactions
        if e['sad'] > 30 or e['fear'] > 30:
            return "2_Candids/Emotional_Deep" # Pensive, moody, or artistic sorrow
            
        return "3_Review_Required/Mixed_Expression"

    except:
        # All non-human/unrecognized shots
        return "non face recognised images"

st.title("📸 Pro-Photo Culling Engine")
st.write("Sorting by nuance: Laughs, Subtle Smiles, Editorial Focus, and Candids.")

# No manual reset needed - form clears itself on new upload
with st.form("pro_batch_form", clear_on_submit=True):
    files = st.file_uploader("Upload Session", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    process_btn = st.form_submit_button("🚀 Analyze & Sort Batch")

if process_btn and files:
    zip_buffer = io.BytesIO()
    progress_bar = st.progress(0)
    status = st.empty()
    
    with zipfile.ZipFile(zip_buffer, "a") as zip_f:
        for i, file in enumerate(files):
            # Progress tracking
            status.text(f"Processing {i+1}/{len(files)}: {file.name}")
            progress_bar.progress((i + 1) / len(files))
            
            img_bytes = file.read()
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
            
            cat = get_pro_category(img)
            zip_f.writestr(f"{cat}/{file.name}", img_bytes)
            
    status.success(f"✅ Processed {len(files)} images.")
    st.download_button("📂 Download Sorted ZIP", zip_buffer.getvalue(), "Pro_Cull_Result.zip")
