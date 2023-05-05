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
