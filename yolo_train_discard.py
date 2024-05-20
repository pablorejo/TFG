from ultralytics import YOLO
from globales import RUTA_DESTINO_TRAINING,NOMBRE_ARCHIVO_BUENAS,NOMBRE_ARCHIVO_MALAS,IMGSZ
from funciones import vaciar_carpeta, copiar_a_training_file
vaciar_carpeta(RUTA_DESTINO_TRAINING)
copiar_a_training_file('buenas',NOMBRE_ARCHIVO_BUENAS)
copiar_a_training_file('malas',NOMBRE_ARCHIVO_MALAS)

model = YOLO('yolov8n-cls.pt')
results = model.train(epochs=3, imgsz=IMGSZ)