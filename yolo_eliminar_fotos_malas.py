from ultralytics import YOLO
import os
import globales
import pandas as pd

from PIL import Image

# Carga el modelo YOLO preentrenado
model_path = 'runs/classify/train2/weights/best.pt'  # Cambia esto por la ruta de tu modelo
model = YOLO(model_path)

# Define la ruta de la imagen a clasificar
image_paths = globales.encontrar_imagenes_jpg(globales.carpeta_de_imagenes) # Cambia esto por la ruta de tu imagen

for image_path in image_paths:
    
    # Realiza la predicción en la imagen
    results = model(image_path)
    
    # Suponiendo que `results` es una lista de objetos `Results`
    for result in results:
        # Accede a las cajas delimitadoras, confianzas, y clases
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs

        # Obtiene el índice de la clase top 1 (la más probable)
        top1_class_index = probs.top1

        # Obtiene la confianza de la clase top 1
        top1_confidence = probs.top1conf

        # Obtiene los índices de las top 5 clases
        top5_class_indices = probs.top5

        # Obtiene las confidencias de las top 5 clases
        top5_confidences = probs.top5conf

        if (result.names[top1_class_index] == globales.tipos['mala']):
            if os.path.isfile(image_path) or os.path.islink(image_path):
                
                if (top1_confidence < 0.75):
                    image = Image.open(image_path)
                    # Mostrar la imagen
                    image.show()
                    print(f"Clase más probable: {result.names[top1_class_index]} con confianza {top1_confidence}\n")
                    respuesta = input('¿Desea eliminar la imagen? (s/n)\n')
                    if (respuesta == 's'):
                        os.remove(image_path) 
                else:
                    os.remove(image_path)
                    