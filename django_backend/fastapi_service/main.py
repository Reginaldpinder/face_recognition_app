from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from deepface_local.deepface import DeepFace

app = FastAPI()

@app.post("/compare/")
async def compare_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    img1 = np.frombuffer(image1.file.read(), np.uint8)
    img2 = np.frombuffer(image2.file.read(), np.uint8)
    
    img1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(img2, cv2.IMREAD_COLOR)

    result = DeepFace.verify(img1, img2, model_name="Facenet")

    return {"match": result["verified"], "confidence": result["distance"]}
