import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd
import os
from DPP_Filters import depth_filter
import json
from scipy import stats
from ahrs.filters import Madgwick,EKF, Mahony
import open3d as o3d
import threading
from plot3d import create_arrow,create_bidirectional_white_axes,create_arrow_cap 
from collections import deque
import supervision as sv
import copy




#Function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

#Global variables and constants

sma_window_size = 10  # Avereging filter window size
depth_buffer = {}  # Per track_id buffer for averaging filter
fps = []  # List to calculate FPS possible

tracker=sv.ByteTrack()
#camera parameters and configurations

def camera_config_1():
    ''' CAMERA CONFIGURATION'''
    # Intel RealSense camera pipeline configuration
    pipeline = rs.pipeline()
    config = rs.config()

    # Load JSON configuration
    device = rs.context().devices[0]
    advanced_mode = rs.rs400_advanced_mode(device)

    if not advanced_mode.is_enabled():
        print("Enabling advanced mode...")
        advanced_mode.toggle_advanced_mode(True)



    # Load JSON config file
    with open("camera_preset.json") as file:
        json_text = file.read()
        advanced_mode.load_json(json_text)
        print("✅ JSON configuration applied.")


    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)
    pipeline.start(config)

    sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    if sensor.supports(rs.option.emitter_enabled):
        sensor.set_option(rs.option.emitter_enabled, 0)
        print("IR Emitter disabled.")
    else:
        print("Emitter option not supported on this device.")

    if sensor.supports(rs.option.laser_power):
        sensor.set_option(rs.option.laser_power, 0)
        print("Laser power set to 0.")


    align = rs.align(rs.stream.color)
    return align, pipeline


    '''camera configuration ends'''


def camera_config_2(file_path):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(file_path)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return align,pipeline




#Gpu configuration and model loading

Yolo_path = r"model/cus_dat_v11s.pt"   #select the model path
device = "cuda" if torch.cuda.is_available() else "cpu"                                                                             #check if cuda is available
torch.cuda.set_per_process_memory_fraction(0.125, device=0)                                                                         #set the gpu memory fraction to 0.125, access to only 1/8 of the gpu memory 
torch.set_num_threads(4)                                                                                                            #set the number of threads to 4    
torch.set_num_interop_threads(2)                                                                                                    #set the number of interop threads to 2 
model = YOLO(Yolo_path, task='detect')                                                                                              #load the model                                              

#Processing Pipeline

def main(align, pipeline,target_class):
    
    madgwick = Mahony()
    q = np.array([1.0, 0.0, 0.0, 0.0])
    prev_time = time.time()


    

    t= np.array([0, 0, 0])

    # DataFrame for data output
    df = pd.DataFrame(columns=["Timestamp", "Object_ID", "Depth_mm"])
    csv_filename = "depth_tracking.csv"

    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)


    x_history = []
    y_history = []
    z_history = []


    t1=t2=0


    # Setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Real-time 3D Plot', width=800, height=600)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])

    # Axes and arrow caps
    axis = create_bidirectional_white_axes(length=1.0)
    axis.scale(1.5, center=axis.get_center())
    vis.add_geometry(axis)


    # Correctly aligned arrow caps at +X, +Y, +Z
    x_arrow = create_arrow_cap('x', [1.05, 0, 0],color=[1, 0, 0])
    y_arrow = create_arrow_cap('y', [0, 1.05, 0],color=[0, 1, 0])
    z_arrow = create_arrow_cap('z', [0, 0, 1.05],color=[0, 0, 1])

    for arrow in [x_arrow, y_arrow, z_arrow]:
        vis.add_geometry(arrow)


    original_arrow = create_arrow(length=1)
    arrow = copy.deepcopy(original_arrow)
    vis.add_geometry(arrow)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector([])
    pcd.colors = o3d.utility.Vector3dVector([])
    vis.add_geometry(pcd)
    history = []


    try:
        start_time = time.time()
        target_track=-1
        nil_count=0
        while True:
            
            frames = pipeline.wait_for_frames()
            t1=time.time()
            frames = align.process(frames)
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            acc_data = accel_frame.as_motion_frame().get_motion_data()
            gyr_data = gyro_frame.as_motion_frame().get_motion_data()
            acc = np.array([acc_data.x, acc_data.y, acc_data.z])
            gyr = np.array([gyr_data.x, gyr_data.y, gyr_data.z])

            now = time.time()
            dt = now - prev_time
            prev_time = now
            madgwick.Dt = dt

            q = madgwick.updateIMU(q=q, gyr=gyr, acc=acc)

            R = quaternion_to_rotation_matrix(q)


            if not depth_frame or not color_frame:
                continue

            #depth_frame = depth_filter(depth_frame)

            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())


            #kernel = np.ones((5, 5), np.uint16)
            #depth_image = cv2.dilate(depth_image, kernel, iterations=1)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            results = model.predict(color_image, device=0, conf=0.7)   
            detections = sv.Detections.from_ultralytics(results[0])
            tracked_detections = tracker.update_with_detections(detections)  

            annotated_frame = results[0].plot()



            data_list = []
            count = 0
            if results[0].boxes is not None:
                for [x1,y1,x2,y2], conf, class_id, track_id in zip(
                    tracked_detections.xyxy,
                    tracked_detections.confidence,
                    tracked_detections.class_id,
                    tracked_detections.tracker_id
                    ): 

                    if int(class_id) != target_class :
                        continue 

                    if target_track == -1:
                        target_track =track_id
                    if track_id != target_track:
                        continue
                    else:
                        count+=1
                    #print(conf)

                    y1, y2 = int(y1), int(y2)
                    x1, x2 = int(x1), int(x2)


                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)


                    p = depth_image[y1:y2, x1:x2]
                    raw_depth = np.median(p[p != 0 ])#*1.5


                    
                    # Initialize buffer for track_id if not already
                    if track_id not in depth_buffer:
                        depth_buffer[track_id] = deque(maxlen=sma_window_size)

                    # Append new raw depth to buffer
                    depth_buffer[track_id].append(raw_depth)

                    # Apply moving average filter
                    actual_depth = np.nanmean(depth_buffer[track_id])/10

                    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], actual_depth/100)
                    #print(f"3D coordinates at ({cx},{cy}): {point_3d}: {actual_depth}" )
                    
                    point_3d = R@point_3d + t
                    
                    # Remove previous arrow geometry from visualizer
                    
                    

                    
                    data_list.append([time.time()-start_time, int(class_id), point_3d[0],point_3d[1],point_3d[2]])

                    
                    history.append(point_3d)
                    if len(history) > 10:
                        history.pop(0)
        

                    pcd.points = o3d.utility.Vector3dVector(history)
                    colors = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(history), 1))  # Green
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                    x_history.append(point_3d[0])
                    y_history.append(point_3d[2])
                    z_history.append(point_3d[1])

                    cv2.putText(annotated_frame,
                                f"Depth: {actual_depth} cm",
                                (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2)
            else:
                nil_count+=1


            if count<1:
                nil_count+=1
                if nil_count>10:
                    target_track=-1
                    nil_count=0
    

            if data_list:
                new_data = pd.DataFrame(data_list, columns=["Timestamp", "Object_ID", "x_coordinate","y_coordinate","z_coordinate"])
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(csv_filename, index=False)
            cv2.imshow("frame", annotated_frame)

            vis.remove_geometry(arrow, reset_bounding_box=False)

            # Copy a fresh arrow from the original
            arrow = copy.deepcopy(original_arrow)

            # Create the transformation matrix (rotation + translation)
            T = np.eye(4)
            T[:3, :3] = R       # Rotation from IMU
            T[:3, 3] = t        # Translation offset

            # Apply the transformation
            arrow.transform(T)


            vis.update_geometry(pcd)
            # Add the transformed arrow back to visualizer
            vis.add_geometry(arrow, reset_bounding_box=False)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)
            t2=time.time() 
            fps.append(1/(t2-t1))
                    




    except KeyboardInterrupt:
        print("Stopped by user.")  
        print(np.mean(fps))      

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()



if __name__ == '__main__':
    x=input("Bag File?(Y/N/Q-exit)")
    while True:
        if x=='y' or x=='Y':
            file_path=input("file Path:")
            align,pipeline=camera_config_2(file_path)
            break
        elif x=='N' or x=='n':
            align,pipeline=camera_config_1()
            break
        elif x=='q' or x=='Q':
            exit()
    target_class = input("target_class:")
    try:
        target_class = int(target_class)
    except ValueError:
        print("Invalid target_class input. Using default (0).")
        target_class = 0
    print(target_class)
    if target_class in [0,1,2,3,4,5,6,56]:   
        main(align,pipeline,target_class)
    else:
        main(align,pipeline)

    


