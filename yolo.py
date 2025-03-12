import cv2 
from ultralytics import YOLO
import pyrealsense2 as rs 
import numpy as np

# Load the YOLOv11 model 
model = YOLO("yolo11n.pt")  

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Access webcam 
pipeline.start(config)
try:
    while True:
        # Visualize the detected objects on the frame
        #annotated_frame = results[0].render() 

        frames = pipeline.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        #if not depth_frame:
            #continue
        
        
        results = model.predict(frames, imgsz=(640, 480))
        
        
        # Convert to numpy array
        #depth_image = np.asanyarray(depth_frame.get_data())

        # Get depth at a specific pixel (center of the image)
        #height, width = depth_image.shape
        #center_x, center_y = width // 2, height // 2
        #depth_mm = depth_frame.get_distance(center_x, center_y) * 1000  # Convert meters to mm

        # Show depth image
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #cv2.circle(depth_colormap, (center_x, center_y), 5, (0, 255, 0), -1)
        #cv2.putText(depth_colormap, f"Depth: {depth_mm:.0f} mm", (center_x - 50, center_y - 10),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        #cv2.imshow("Depth Image", depth_colormap)

        results.show()

        
        #cv2.imshow('YOLOv11 Realtime', frame) 


        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()