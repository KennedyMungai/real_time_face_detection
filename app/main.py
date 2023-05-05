"""The entrypoint to the application"""
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()
cascade_classifier = cv2.CascadeClassifier()


class Faces(BaseModel):
    """The model for the faces"""
    faces: List[Tuple[int, int, int, int]]


@app.get("/", name="Home", description="The home endpoint for the backend application", tags=['Home'])
async def home() -> dict[str, str]:
    """The home endpoint for the backend application

    Returns:
        dict[str, str]: _description_
    """
    return {"Hello": "World"}


@app.post("/face-detection", response_model=Faces)
async def face_detection(image: UploadFile = File(...)) -> Faces:
    """An endpoint for uploading photos for facial detection

    Args:
        image (UploadFile, optional): The image uploaded to the server. Defaults to File(...).

    Returns:
        Faces: The template for the Face data
    """
    data = np.fromfile(image.file, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray)

    if len(faces) > 0:
        faces_output = Faces(faces=faces.tolist())
    else:
        faces_output = Faces(faces=[])

    return faces_output
