from ultralytics import YOLO
from globales import *

if __name__ == "__main__":
    recortarImagenes(src_img="1055938935.jpg", model=YOLO('yolo_detect.pt'))