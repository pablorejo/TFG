import globales
from ultralytics import YOLO




for ruta in globales.ruta.values():
    globales.vaciar_carpeta(ruta)

# for categori_taxonomica in globales.rangos_taxonomicos:
    
# Entrenar dependiendo todo de golpe
directorios = globales.obtener_directorios(globales.carpeta_de_imagenes)

for clase in directorios:
    imagenes = globales.encontrar_imagenes_jpg(f'{globales.carpeta_de_imagenes}/{clase}')
    globales.copiar_a_training(tipo=clase,lineas=imagenes)

model = YOLO('yolov8n-cls.pt') 
results = model.train(data='training.yaml', epochs=1, imgsz=256, name='clases')
