from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov8n-cls.pt') 
    results = model.train(data='training', epochs=20, imgsz=256)