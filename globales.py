import os, platform, random
import shutil #Para copiar las imagenes en la carpeta de entrenamiento
import random,math
from PIL import Image,UnidentifiedImageError
from imgaug import augmenters as iaa
import imageio
import numpy as np
from ultralytics import YOLO
import pandas as pd
import cv2
from procesar_pandas import procesar_pandas

VERBOSE = False #En caso de que se quiera que aparezca texto durante las ejecuciones
def warning(text: str):
    if VERBOSE:
        print(Colors.WARNING, text, Colors.ENDC)

def fail(text: str):
    print(Colors.FAIL, text, Colors.ENDC)

def info(text: str):
    if VERBOSE:
        print(Colors.OKGREEN, text, Colors.ENDC)
    
# Ejemplo de secuencias de escape ANSI para diferentes colores
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m WARNING: '
    FAIL = '\033[91m ERROR: '
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
DESORDENAR_DATAFRAME = False # Esto indida si se tiene que desordenar el dataframe o no, es útil para que obtenga los datos aleatoriamente del dataframe.
ENTRENAR_LOCAL = False #Esto quiere decir que se va a entrenar de manera local con las imagenes que están ya descargadas en la carpeta definida glovales.carpeta_de_imagenes
RUTA_IMG_TEMPORALES = "img_temporales" # En esta carpeta se van a descargar temporalmente las imagenes para despues moverlas a la carpeta de entrenamiento
RUTA_IMG_DETECCION = "imagenes_deteccion"

MODEL_DESCARTAR_PATH = 'yolo_modelos/yolo_discard.pt' 
if(os.path.exists(MODEL_DESCARTAR_PATH)): 
    model_descartar = YOLO(MODEL_DESCARTAR_PATH)
else:
    warning(f"No existe el modelo para descartar imagenes: {MODEL_DESCARTAR_PATH}\nNo se van a descartar las imagenes malas")
    
    
MODEL_DETECT_PATH = 'yolo_modelos/yolo_detect.pt'
if(os.path.exists(MODEL_DETECT_PATH)): 
    model_detect = YOLO(MODEL_DETECT_PATH)
else:
    model_detect = None
    warning(f"No existe el modelo para detectar imagenes: {MODEL_DETECT_PATH}\nNo se van a detectar las imagenes")
    
RUTA_PANDAS_CSV = 'pandas_ficheros'
CSV_DATOS = os.path.join(RUTA_PANDAS_CSV,'ocurrencias_parseado.csv')
CSV_DATOS_TODAS = os.path.join(RUTA_PANDAS_CSV,'ocurrencias_parseado_todas.csv') #Todas las imagenes independiente mente si son buenas o malas

if(not os.path.exists(CSV_DATOS)):
    fail(f"No existe el fichero {CSV_DATOS} es necesario crear el fichero con los datos en formato csv")
    print("""Que desea hacer?
1) Usar el fichero de datos csv todos como el fichero procesado
2) Cancelar
3) Continuar igualmente""")
    numero = int(input(": "))
    if (os.path.exists(CSV_DATOS_TODAS)):
        if numero == 1:
            CSV_DATOS == CSV_DATOS_TODAS
        elif numero == 2:
            exit(0)
        else:
            pass
    else:
        fail(f"No existe el fichero {CSV_DATOS_TODAS} es necesario crear el fichero con los datos en formato csv\nSaliendo del programa...")
        exit(-1)

Image.MAX_IMAGE_PIXELS = None # Permite que no tenga limite de numero maximo de pixeles.

IMGSZ = 640
EPOCAS_DE_ENTRENAMIENTO = 2

# Ruta a la carpeta donde se encuentran todas las imaganes
CARPETA_IMAGENES = 'imagenes'

# Tipos de imagenes en la primera parte del entrenamiento, buenas y malas
RUTA_TXT_DISCART = "ficheros_txt"
NOMBRE_ARCHIVO_BUENAS = os.path.join(RUTA_TXT_DISCART,'imagenes_buenas.txt')
NOMBRE_ARCHIVO_MALAS = os.path.join(RUTA_TXT_DISCART,'imagenes_buenas.txt')
tipos = {
    'buena': 'buenas',
    'mala': 'malas'
}


# Ruta donde se va van a guardar los datos de entrenamiento
RUTA_DESTINO_TRAINING = 'datasets/imagenet10'
ruta_training_data = {
    'train': os.path.join(RUTA_DESTINO_TRAINING,'train'),
    'test': os.path.join(RUTA_DESTINO_TRAINING,'test'),
    'valid': os.path.join(RUTA_DESTINO_TRAINING,'valid'),
}

RUTA_DESTINO_TRAINING_DETECT = 'datasets/detect'
ruta_training_detect_data = {
    'train': os.path.join(RUTA_DESTINO_TRAINING_DETECT,'train'),
    'test': os.path.join(RUTA_DESTINO_TRAINING_DETECT,'test'),
    'valid': os.path.join(RUTA_DESTINO_TRAINING_DETECT,'valid'),
}

# Los rangos taxonomicos que existen y su nivel de recursividad en las carpetas para realizar el entrenamiento recursivo
RANGOS_TAXONOMICOS = [
    ('clase',1),
    ('orden',2),
    ('familia',3),
    ('genero',4)
    # ,('especie',5)
]

# Para el entrenamiento se establece aquí los distintos porcentajes para el entrenamiento la validacion y el testeo de la aplicacion.
PORCENTAJE_DE_VALIDACION = 0.1
PORCENTAJE_DE_TESTING = 0.02
PORCENTAJE_DE_TRAINING = 1 - PORCENTAJE_DE_TESTING - PORCENTAJE_DE_VALIDACION
NUMERO_DE_MUESTRAS_IMAGEN = 3 # Esto son el numero de imagenes que se tendran por cada clase distinta como maximo, si no se llega hacemos 
if (NUMERO_DE_MUESTRAS_IMAGEN < 3):
    fail(f"Tiene que haber al menos 3 imagenes por cada categoria")
    exit(-1)

CONF_TOP_5 = 0.9