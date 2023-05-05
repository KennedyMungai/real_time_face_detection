"""The entrypoint to the application"""
import asyncio
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
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


@app.on_event("startup")
async def startup():
    """The startup event"""
    cascade_classifier.load(cv2.data.haarcascades +
                            "haarcascade_frontalface_default.xml")


async def receive(websocket: WebSocket, queue: asyncio.Queue):
    """A function for queuing the data from the websocket

    Args:
        websocket (WebSocket): The websocket
        queue (asyncio.Queue): The queue
    """
    bytes = await websocket.receive_bytes()

    try:
        queue.put_nowait(bytes)
    except asyncio.QueueFull:
        pass


async def detect(websocket: WebSocket, queue: asyncio.Queue):
    """The face detection code from the backend queue

    Args:
        websocket (WebSocket): The websocket object
        queue (asyncio.Queue): The image queue
    """
    while True:
        bytes = await queue.get()
        data = np.frombuffer(bytes, dtype=np.uint8)
        img = cv2.imdecode(data, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade_classifier.detectMultiScale(gray)

        if len(faces) > 0:
            faces_output = Faces(faces=faces.tolist())
        else:
            faces_output = Faces(faces=[])

        await websocket.send_json(faces_output.dict())


@app.websocket("/face-detection")
async def face_detection_websocket(websocket: WebSocket):
    """The face detection websocket

    Args:
        websocket (WebSocket): The websocket
    """
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    detect_task = asyncio.create_task(detect(websocket, queue))

    try:
        while True:
            await receive(websocket, queue)
    except WebSocketDisconnect:
        await websocket.close()
