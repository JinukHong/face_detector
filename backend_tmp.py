from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        buffer.write(file.file.read())

    # Process the image (convert to grayscale for simplicity)
    img = cv2.imread(file_location)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_file_location = os.path.join(UPLOAD_DIR, f"processed_{file.filename}")
    cv2.imwrite(processed_file_location, gray_img)

    return JSONResponse(content={"message": "File uploaded and processed successfully!", "processed_file": f"processed_{file.filename}"})

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
