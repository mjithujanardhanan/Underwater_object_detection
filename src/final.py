import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pandas as pd
import os


device= "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt").to(device)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

df = pd.DataFrame(columns=["Timestamp", "Object_ID", "Depth_mm"])
csv_filename = "depth_tracking.csv"

if os.path.exists(csv_filename):
    df = pd.read_csv(csv_filename)

depth_history = []
time_history = []

fig, ax = plt.subplots()
ax.set_ylim(0, 3000) 
ax.set_xlim(0, 50)  
line, = ax.plot([], [], 'g-', lw=2)

def update_plot(frame):
    if not time_history:  
        return line,

    if len(time_history) > 50:  
        depth_history.pop(0)
        time_history.pop(0)

    line.set_data(time_history, depth_history)
    
    ax.set_xlim(max(0, time_history[0] - 1), time_history[-1] + 1)
    return line,


ani = animation.FuncAnimation(fig, update_plot, interval=500, blit=True)



try:
    start_time = time.time()
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        depth_image= np.asanyarray(depth_frame.get_data())
        color_image= np.asanyarray(color_frame.get_data())

        kernel = np.ones((5, 5), np.uint16)

        depth_image = cv2.dilate(depth_image, kernel, iterations=1)

        results = model.track(color_image, persist=True)
        annotated_frame = results[0].plot()
        
        data_list = []

        if results[0].boxes is not None:
            for box in results[0].boxes.data.cpu().numpy():
                if box.shape[0] == 7:  
                    x1, y1, x2, y2, conf, cls, track_id = box
                else:  
                    x1, y1, x2, y2, conf, cls = box
                    track_id = -1  

                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                depth_value = depth_image[cy,cx]
                depth_value = depth_value

                depth_history.append(depth_value)
                time_history.append(time.time() - start_time)
                data_list.append([time.time(), int(track_id), depth_value])

                cv2.putText(annotated_frame,
                            f"Depth: {depth_value} cm",
                            (cx,cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0,255,0),
                            2,
                            )
                print(f"Object {int(cls)} - Track ID {int(track_id)}: Depth = {depth_value} mm")

        if data_list:
            new_data = pd.DataFrame(data_list, columns=["Timestamp", "Object_ID", "Depth_mm"])
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(csv_filename, index=False)

        cv2.imshow("frame", annotated_frame)

        plt.pause(0.001)
        #print(depth_image)
        #break

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break





finally:
    pipeline.stop()
    cv2.destroyAllWindows()