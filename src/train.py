from ultralytics import YOLO


# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights


results = model.train(data=r"C:\Users\jithu\Desktop\cameraenv\cameraenv\tub data.v10i.yolov5pytorch\data.yaml",
                      epochs=20, imgsz=640, batch=32, device=0, workers=12)










