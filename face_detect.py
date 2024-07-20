import av
import cv2
import dlib
import csv
import time
from datetime import datetime
import streamlit as st

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# dlib 얼굴 검출기 초기화
detector = dlib.get_frontal_face_detector()

# Streamlit UI 컴포넌트
flip = st.checkbox("Flip")

# CSV 파일을 열고, 헤더를 작성합니다.
csv_file = open('detected_faces.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['timestamp', 'face_x1', 'face_y1', 'face_x2', 'face_y2'])

# FPS 계산을 위한 변수 초기화
frame_count = 0
start_time = time.time()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global frame_count, start_time
    img = frame.to_ndarray(format="bgr24")

    # 이미지 좌우 반전
    if flip:
        img = img[:, ::-1, :]
        print("Flipping the video frame")

    # dlib을 사용한 얼굴 인식
    faces = detector(img, 1)  # 이미지에서 얼굴을 찾습니다.
    print(f"Detected {len(faces)} faces")

    if len(faces) > 0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            print(f"Face detected at [(x1: {x1}, y1: {y1}), (x2: {x2}, y2: {y2})]")
            writer.writerow([timestamp, x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

    print("Run completed")
    return av.VideoFrame.from_ndarray(img, format="bgr24")


class MyVideoTransformer(VideoTransformerBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        processed_image = video_frame_callback(frame)
        return av.VideoFrame.from_ndarray(processed_image, format="bgr24")


# WebRTC 스트리머 설정
webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    video_processor_factory=MyVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
