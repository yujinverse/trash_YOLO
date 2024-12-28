from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import torch
import numpy as np

app = FastAPI()

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5m.pt', force_reload=True)

# 카메라 초기화
camera = cv2.VideoCapture(0)  # 0번 카메라는 기본 웹캠


def generate_frames():
    """카메라에서 프레임을 가져와 YOLOv5로 처리하고 반환"""
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # YOLOv5 모델로 프레임 처리
        results = model(frame)

        # 결과를 프레임에 렌더링
        annotated_frame = np.squeeze(results.render())

        # 프레임을 JPEG 형식으로 인코딩
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # 프레임을 yield
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/video-stream")
def video_stream():
    """카메라 스트림을 클라이언트에 전달"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/")
def read_root():
    return {"message": "YOLOv5 Real-Time Object Detection Server is Running!"}
