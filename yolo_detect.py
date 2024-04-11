from ultralytics import YOLO
from globales import *

if __name__ == "__main__":
    recortarImagenes(src_img="prueba.png", model=YOLO('yolov8n.pt'))