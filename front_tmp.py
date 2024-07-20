import av
import cv2
import dlib
import csv
import time
from datetime import datetime
import requests
import numpy as np
from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO


# dlib 얼굴 검출기 초기화
# detector = dlib.get_frontal_face_detector()

def model_load():
    return YOLO(model="model/yolov8n-face.onnx", task="detect", verbose=False)

detector = model_load()

# Streamlit UI 컴포넌트
flip = st.checkbox("Flip")

# CSV 파일을 열고, 헤더를 작성합니다.
csv_file = open('detected_faces.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['timestamp', 'face_x1', 'face_y1', 'face_x2', 'face_y2'])

# FPS 계산을 위한 변수 초기화
frame_count = 0
start_time = time.time()

def send_image_to_server(image, filename='face.jpg'):
    _, buffer = cv2.imencode('.jpg', image)
    response = requests.post("http://localhost:8000/upload/", files={"file": buffer.tobytes()})
    return response.json()

def process_image(img):
    global frame_count, start_time

    # 이미지 좌우 반전
    if flip:
        img = img[:, ::-1, :]
        print("Flipping the image")

    # dlib을 사용한 얼굴 인식
    faces = detector(img, verbose=False)
    print(f"Detected {len(faces)} faces")

    if len(faces) > 0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            print(f"Face detected at [(x1: {x1}, y1: {y1}), (x2: {x2}, y2: {y2})]")
            writer.writerow([timestamp, x1, y1, x2, y2])

            # Crop and send the face image to the server
            cropped_face = img[y1:y2, x1:x2]
            result = send_image_to_server(cropped_face)
            return result["processed_file"]

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

    print("Run completed")
    return None

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    st.session_state["last_frame"] = img
    return av.VideoFrame.from_ndarray(img, format="bgr24")

class MyVideoTransformer(VideoTransformerBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        st.session_state["last_frame"] = image
        return av.VideoFrame.from_ndarray(image, format="bgr24")

st.title("Image Upload and Webcam Processing")

# Layout with two columns
left_column, right_column = st.columns(2)

# Initialize session state for processed file and last frame
if "processed_file" not in st.session_state:
    st.session_state["processed_file"] = None
if "last_frame" not in st.session_state:
    st.session_state["last_frame"] = None

# Left column for image upload or webcam
with left_column:
    upload_choice = st.radio("Choose an option:", ("Upload an image", "Use webcam"))

    if upload_choice == "Upload an image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, channels="BGR")
            processed_file = process_image(img)
            if processed_file:
                st.session_state["processed_file"] = processed_file
    else:
        webrtc_streamer(
            key="example",
            video_frame_callback=video_frame_callback,
            video_processor_factory=MyVideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        if st.button("Capture Image"):
            if st.session_state["last_frame"] is not None:
                img = st.session_state["last_frame"]
                st.image(img, channels="BGR")
                processed_file = process_image(img)
                if processed_file:
                    st.session_state["processed_file"] = processed_file

# Display the processed image in the right column
with right_column:
    if st.session_state["processed_file"]:
        st.image(f"http://localhost:8000/uploads/{st.session_state['processed_file']}", caption="Processed Image")
    else:
        # 뭔가 UI상으로는 채워두는게 필요
        st.write("processed_image")
# Ensure the CSV file is closed properly on app exit
st.write("Application running...")
csv_file.close()
