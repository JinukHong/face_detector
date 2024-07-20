import av
import cv2
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import requests

# https://github.com/fullmakeralchemist/gitstreamlit/blob/master/helper.py
@st.cache_resource
def model_load():
    return YOLO(model="model/yolov8n-face.onnx", task="detect", verbose=False)

model = model_load()
flip = st.checkbox("Flip")

def send_image_to_server(image, filename='detected_face.jpg'):
    _, buffer = cv2.imencode('.jpg', image)
    response = requests.post("http://localhost:8000/upload/", files={"file": (filename, buffer.tobytes(), "image/jpeg")})
    print(f"Server response: {response.text}")


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    flipped = img[:,::-1,:] if flip else img
    # resize 하고 추론 시간 테스트 할 것
    results = model(flipped, verbose=False)
    if len(results) > 1:
        print("1")
        message = "Please ensure only one person is in the frame."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255)  # Red color
        thickness = 2
        text_size, _ = cv2.getTextSize(message, font, font_scale, thickness)
        text_x = (flipped.shape[1] - text_size[0]) // 2
        text_y = (flipped.shape[0] + text_size[1]) // 2
        cv2.putText(flipped, message, (text_x, text_y), font, font_scale, color, thickness)
    elif len(results) == 1:
        # yolov8 process 진행
        print("2")
        res = results[0]
        x1, y1, x2, y2 = res.boxes.xyxy
        print(x1, y1, x2, y2)
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(flipped, (x1, y1), (x2, y2), color, thickness)
        cropped_face = flipped[int(y1):int(y2), int(x1):int(x2)]
        send_image_to_server(cropped_face)


    else:
        print("nothing")
    print("run")
        
    return av.VideoFrame.from_ndarray(flipped, format="bgr24")


class MyVideoTransformer(VideoTransformerBase):
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        processed_image = self._display_detected_frames(image)
        # st.image(processed_image, caption='Detected Video', channels="BGR", use_column_width=True)
        # 위 코드는 main thread충돌 때문에 아래코드 사용 필요
        return av.VideoFrame.from_ndarray(processed_image, format="bgr24")

    def _display_detected_frames(self, image):
        orig_h, orig_w = image.shape[0:2]
        width = 720  # Set the desired width for processing

        # cv2.resize used in a forked thread may cause memory leaks
        input = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))

        if self.model is not None:
            # Perform object detection using YOLO model
            res = self.model.predict(input, conf=self.conf)
            # x1, y1, x2, y2 = res.boxes.xyxy
            # print(x1, y1, x2, y2)
            print(res[0][0])
            # if res.boxes is not None:
            #     for box in res.boxes:
            #         x1, y1, x2, y2 = map(int, [box.x1, box.y1, box.x2, box.y2])
            #         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #         # Crop the detected face
            #         cropped_face = image[y1:y2, x1:x2]
            #         if cropped_face.size > 0:
            #             self.send_image_to_server(cropped_face)

            # Plot the detected objects on the video frame
            res_plotted = res[0].plot()
            return res_plotted

        return input

    def send_image_to_server(self, image, filename='detected_face.jpg'):
        _, buffer = cv2.imencode('.jpg', image)
        response = requests.post("http://localhost:8000/upload/", files={"file": (filename, buffer.tobytes(), "image/jpeg")})
        if response.status_code == 200:
            print("Image sent successfully.")
        else:
            print(f"Failed to send image: {response.status_code}, {response.text}")

webrtc_streamer(
    key="example",
    # video_frame_callback=video_frame_callback,
    video_processor_factory=lambda: MyVideoTransformer(0.5, model),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)