from ultralytics import YOLO
from globales import *
copiar_a_training_deteccion('imagenes_deteccion')
model = YOLO('yolov8n.pt')
results = model.train(data='conf.yaml', epochs=30, imgsz=IMGSZ)

