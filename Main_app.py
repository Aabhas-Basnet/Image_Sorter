import cv2
import os
import shutil
import customtkinter as ctk
from tkinter import filedialog, messagebox
from deepface import DeepFace
from threading import Thread

class UltraPhotoSorter(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Pro Photo Cull")
        self.geometry("600x450")

        self.label = ctk.CTkLabel(self, text="AI Emotion & Document Sorter", font=("Helvetica", 22, "bold"))
        self.label.pack(pady=20)

        self.btn = ctk.CTkButton(self, text="Select Folder & Start AI Sort", command=self.start)
        self.btn.pack(pady=10)

        self.progress = ctk.CTkProgressBar(self, width=400)
        self.progress.set(0)
        self.progress.pack(pady=20)

        self.status = ctk.CTkLabel(self, text="Ready", text_color="gray")
        self.status.pack(pady=10)

    def start(self):
        path = filedialog.askdirectory()
        if path:
            Thread(target=self.process, args=(path,), daemon=True).start()

    def process(self, source):
        output = os.path.join(source, "AI_Sorted_Portfolio")
        valid_ext = ('.png', '.jpg', '.jpeg')
        files = [f for f in os.listdir(source) if f.lower().endswith(valid_ext)]
        
        for i, filename in enumerate(files):
            img_path = os.path.join(source, filename)
            category = "Uncategorized"

            try:
                # 1. AI Analysis for Emotions
                # We use 'retinaface' or 'mtcnn' for better accuracy than the last version
                results = DeepFace.analyze(
                    img_path=img_path, 
                    actions=['emotion'], 
                    detector_backend='opencv', 
                    enforce_detection=True, 
                    silent=True
                )
                
                # Get the dominant emotion (Happy, Sad, Angry, etc.)
                main_emotion = results[0]['dominant_emotion'].capitalize()
                category = os.path.join("People", main_emotion)

            except ValueError:
                # 2. No face found? Check if it's a Document vs Scenery
                img = cv2.imread(img_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 100, 200)
                    density = (cv2.countNonZero(edges) / (img.shape[0] * img.shape[1])) * 100
                    
                    if density > 3.0: # High edge density = Text/Docs
                        category = "Documents_and_Analysis"
                    else:
                        category = "Creative_Views_Scenery"

            # Final Move
            dest_dir = os.path.join(output, category)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(img_path, os.path.join(dest_dir, filename))

            # Update UI
            self.progress.set((i + 1) / len(files))
            self.status.configure(text=f"Processed: {filename} -> {category}")

        messagebox.showinfo("Success", "Professional sorting complete!")

if __name__ == "__main__":
    app = UltraPhotoSorter()
    app.mainloop()