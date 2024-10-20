import uvicorn
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from ultralytics import YOLO

app = FastAPI()

model = YOLO('yolov8n.pt')

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))

        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) 

        results = model(img_array)

        detections = results[0].boxes.data.cpu().numpy()
        class_names = model.names

        predictions = []
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            class_name = class_names[int(cls)]  # Obtener el nombre de la clase usando el índice
            predictions.append({"class_name": class_name,"confidence": float(conf),"coordinates": [float(x1), float(y1), float(x2), float(y2)]})

        return {"predictions": predictions}
    
    except Exception as e:
        return {"error": f"Ocurrió un error: {str(e)}"}

# Ejecutar la aplicación FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)
