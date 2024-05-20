from ultralytics import YOLO
from funciones import copiar_a_training_deteccion
from globales import IMGSZ,RUTA_IMG_DETECCION
import os

def procesar_datos_txt(directorio):
    """
    Procesa los datos de detección de la carpeta para que solamente exista una clase.

    Args:
    directorio (str): La ruta al directorio donde buscar los archivos .txt.

    """
    # Recorrer los archivos en el directorio especificado
    for archivo in os.listdir(directorio):
        # Comprobar si el archivo termina con .txt
        if archivo.endswith('.txt'):
            # Agregar la ruta completa del archivo a la lista
            path = os.path.join(directorio, archivo)
            with open(path,'r') as file_read:
                lineas = file_read.readlines()
                nuevas_lineas = []
                for linea in lineas:
                    if str(linea).split(' ',maxsplit=1)[0] != '0':
                        print(f"la linea {linea} contiene un uno, el archivo es: {path}")
                    nueva_linea = '0 ' + str(linea).split(' ',maxsplit=1)[1]
                    nuevas_lineas.append(nueva_linea)
            
            file_read.close()
            with open(path,'w') as file_write:
                file_write.writelines(nuevas_lineas)
                file_write.close()
                

copiar_a_training_deteccion(RUTA_IMG_DETECCION) # Copiamos los datos a la ruta de entrenamiento
procesar_datos_txt(RUTA_IMG_DETECCION) # Procesamos los datos para que solo exista una clase, en caso de que el usuario se haya confundico
model = YOLO('yolov8n.pt') # Obtenemos el modelo para la clasificicación
results = model.train(data='conf_detect.yaml', epochs=30, imgsz=IMGSZ) # Lo entrenamos con 30 epocas y el archivo de configuracion de conf_detect.yaml
