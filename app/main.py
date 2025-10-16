from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from model_utils import recognize_face

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Face Recognition API running!"}

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = recognize_face(rgb_image)
    return {"result": result}
