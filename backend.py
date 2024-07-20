from fastapi import FastAPI, File, UploadFile, Response, BackgroundTasks
from typing import Tuple
import os
import io
import hashlib
from datetime import datetime

import uvicorn
from PIL import Image, ImageOps

app = FastAPI()

FACE_PATH = "faces"
os.makedirs(FACE_PATH, exist_ok=True)

def save_image(image: Image, path: str) -> None:
    image.save(path)

def generate_filenames() -> Tuple[str, str]:
    
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    random_hash = hashlib.sha256(current_time.encode()).hexdigest()[:8]
    filename = f"{FACE_PATH}/{current_time}_{random_hash}.jpg"
    filename_p = f"{FACE_PATH}/{current_time}_{random_hash}_p.jpg"

    return filename, filename_p
    
@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks):

    # read image
    image = Image.open(io.BytesIO(await file.read()))
    file_name, file_name_p = generate_filenames()
    background_tasks.add_task(save_image, image, f"{file_name}")

    # processing logic
    processed_image = styletransfer(image)

    # return processed image
    buf = io.BytesIO()
    processed_image.save(buf, format="JPEG")
    buf.seek(0)

    # file name : time + hashing 값
    background_tasks.add_task(save_image, processed_image, f"{file_name_p}")

    return Response(content=buf.getvalue(), media_type="image/jpeg")



    
def styletransfer(image: Image) -> Image:
    # color

    # shape

    # color + shape
    
    gray_image = ImageOps.grayscale(image)
    return gray_image




