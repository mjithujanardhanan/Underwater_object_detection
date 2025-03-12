import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np

model = YOLO("yolo11n.pt")

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frame = pipeline.wait_for_frames()
        color_frame = frame.get_color_frame()

        if not color_frame:
            continue
        
        color_image = np.asanyarray(color_frame.get_data()) 

        results = model.track(color_image, persist=True)
        
        annotated_frame = results[0].plot()

        
        cv2.imshow('YOLOv11 Real-Time Object Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()