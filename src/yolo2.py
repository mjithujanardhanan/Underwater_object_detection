import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np

# Load the YOLOv11 model
model = YOLO("yolo11n.pt")

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable RGB stream only
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start RealSense pipeline
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()

        # Extract color frame
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue  # Skip if no valid frame

        # Convert frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # YOLOv11 detection with BOT-SORT tracking
        results = model.predict(color_image, imgsz=(640, 480), tracker='botsort')

        # Annotate and track objects
        annotated_frame = results[0].plot()

        # Add tracking IDs
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            track_id = int(result.id[0]) if result.id is not None else -1  # Tracking ID

            # Annotate ID
            cv2.putText(
                annotated_frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2
            )

        # Display results
        cv2.imshow("YOLOv11 Tracking with IDs", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
