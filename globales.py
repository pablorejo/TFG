import os, platform, random
import shutil #Para copiar las imagenes en la carpeta de entrenamiento
import random,math
from PIL import Image,UnidentifiedImageError
from imgaug import augmenters as iaa
import imageio
import numpy as np
from ultralytics import YOLO
import pandas as pd

DESORDENAR_DATAFRAME = True # Esto indida si se tiene que desordenar el dataframe o no, es útil para que obtenga los datos aleatoriamente del dataframe.

model_path = 'runs/classify/train2/weights/best.pt'  # Cambia esto por la ruta de tu modelo
model = YOLO(model_path)
csv_datos = 'ocurrencias_parseado.csv'

Image.MAX_IMAGE_PIXELS = None #Permite que no tenga limite de numero maximo de pixeles.

imgsz = 256
epocas_entrenamiento = 6

# Ruta a la carpeta donde se encuentran todas las imaganes
carpeta_de_imagenes = 'imagenes'

# Tipos de imagenes en la primera parte del entrenamiento, buenas y malas
nombre_archivo_buenas = 'imagenes_buenas.txt'
nombre_archivo_malas = 'imagenes_malas.txt'
tipos = {
    'buena': 'buenas',
    'mala': 'malas'
}


# Ruta donde se va van a guardar los datos de entrenamiento
ruta_destino_training = 'training'
ruta_training_data = {
    'train': os.path.join(ruta_destino_training,'train'),
    'test': os.path.join(ruta_destino_training,'test'),
    'valid': os.path.join(ruta_destino_training,'valid'),
}

# Los rangos taxonomicos que existen y su nivel de recursividad en las carpetas para realizar el entrenamiento recursivo
rangos_taxonomicos = [
    ('clase',1),
    ('orden',2),
    ('familia',3),
    ('genero',4)
    # ,('especie',5)
]

# Para el entrenamiento se establece aquí los distintos porcentajes para el entrenamiento la validacion y el testeo de la aplicacion.
porcentaje_validacion = 0.1
porcentaje_testing = 0.02
porcentaje_training = 1 - porcentaje_testing - porcentaje_validacion
numero_de_muestras_imagen = 20 # Esto son el numero de imagenes que se tendran por cada clase distinta como maximo, si no se llega hacemos 

def parsear_nombre(nombre):
    reemplazos = {
        ":": "a",
        "<": "b",
        ">": "c",
        "\"": "e",
        "|": "f",
        "?": "g",
        "*": "h"
    }

    nombre = str(nombre)
    for buscar, reemplazar in reemplazos.items():
        nombre = nombre.replace(buscar, reemplazar)
    return nombre

def obtener_carpetas_nivel(ruta, nivel_max, nivel_actual=0):
    """
    Devuelve una lista de carpetas en un cierto nivel de recursividad.

    Args:
    ruta (str): La ruta de inicio.
    nivel_max (int): El nivel específico de recursividad a buscar.
    nivel_actual (int): El nivel actual de recursividad (usado internamente).

    Returns:
    list: Una lista con los caminos relativos a las carpetas en el nivel especificado.
    """
    carpetas = []
    # Comprueba si hemos alcanzado el nivel deseado
    if nivel_actual == nivel_max:
        # Si es un directorio, añade a la lista y retorna
        if os.path.isdir(ruta):
            return [ruta]
        else:
            return []
    
    # Si no estamos en el nivel deseado, busca en los directorios hijos
    if os.path.isdir(ruta):
        with os.scandir(ruta) as entradas:
            for entrada in entradas:
                if entrada.is_dir():
                    # Construye la ruta del directorio hijo
                    ruta_hijo = os.path.join(ruta, entrada.name)
                    # Llama a la función recursivamente y acumula los resultados
                    carpetas += obtener_carpetas_nivel(ruta_hijo, nivel_max, nivel_actual + 1)
    
    return carpetas


def copiar_a_training(tipo, file:str):
    """Copia las imagenes a la carpeta de entrenamiento
    
    Args:
    file: fichero que contine las direcciones de los ficheros a copiar separados por \\n.
    tipo: tipo de dato ejm: 'Bivalvia', 'Caudofaveata'."""
    with open(file) as file:
        lineas = file.read().splitlines()
        random.shuffle(lineas)  # Esto modifica la lista "in-place"
        copiar_a_training(tipo,lineas=lineas)

def copiar_a_training(tipo:str, lineas):
    """Copia las imagenes a la carpeta de entrenamiento
    
    Args:
    lineas: Array[] con las direcciones de los ficheros a copiar.
    tipo: tipo de dato ejm: 'Bivalvia', 'Caudofaveata'."""
    num_valid = math.ceil(len(lineas) * porcentaje_validacion)
    num_test = math.ceil(len(lineas) * porcentaje_testing)
    
    for i in range(num_valid):
        copiar_archivo(lineas[i], f'{ruta_training_data["valid"]}/{tipo}')
    
    for i in range(num_valid,num_valid + num_test):
        copiar_archivo(lineas[i], f'{ruta_training_data["test"]}/{tipo}')
        
    for i in range(num_valid + num_test, len(lineas)):
        copiar_archivo(lineas[i], f'{ruta_training_data["train"]}/{tipo}')

            


def obtener_directorios(ruta_carpeta):
    """Obtiene una lista de los nombres de los directorios (no archivos) dentro de la ruta_carpeta dada."""
    directorios = [nombre for nombre in os.listdir(ruta_carpeta)
    if os.path.isdir(os.path.join(ruta_carpeta, nombre))]
    
    return directorios

def encontrar_imagenes_jpg(directorio: str, num_muestras:int=None):
    """Obtiene todas las imagenes jpg de un directorio en especifico
    directorio: path al directorio del que quieres obtener todas las imagenes
    num_muestras: si pones un valor de num_muestras te devolvera ese total de manera aleatoria.
    """
    imagenes_jpg = []
    n = 0  # Contador total de archivos jpg encontrados
    
    for root, dirs, files in os.walk(directorio):
        for file in files:
            if file.lower().endswith('.jpg'):
                n += 1
                if num_muestras is None:
                    imagenes_jpg.append(os.path.join(root, file))
                # Si aún no hemos recolectado num_muestras imágenes, simplemente añádelas
                elif len(imagenes_jpg) < num_muestras:
                    imagenes_jpg.append(os.path.join(root, file))
                else:
                    # Con probabilidad num_muestras/n, reemplaza un elemento aleatorio
                    s = int(random.random() * n)
                    if s < num_muestras:
                        imagenes_jpg[s] = os.path.join(root, file)
    
    return imagenes_jpg

def encontrar_ficheros(directorio):
    # Lista para almacenar los archivos encontrados
    archivos_encontrados = []

    # Recorre todos los archivos y directorios en el directorio especificado
    for archivo in os.listdir(directorio):
        # Construye la ruta completa del archivo
        ruta_completa = os.path.join(directorio, archivo)
        
        # Verifica si es un archivo
        if os.path.isfile(ruta_completa):
            # Obtiene la extensión del archivo
            _, extension = os.path.splitext(archivo)
            
            # Verifica si la extensión es .txt o .csv
            if extension in ['.txt', '.csv']:
                archivos_encontrados.append(ruta_completa)

    return archivos_encontrados

def apagar_equipo():
    """Apaga el equipo"""
    if platform.system() == "Windows":
        os.system('shutdown /s /t 1')
    elif platform.system() == "Linux":
        os.system('shutdown -h now')
    elif platform.system() == "Darwin":
        os.system('shutdown -h now')  
        
def vaciar_carpeta(ruta_carpeta):
    """Elimina todo lo que esta en una carpeta"""
    for nombre in os.listdir(ruta_carpeta):
        ruta_completa = os.path.join(ruta_carpeta, nombre)
        
        # Verifica si es un archivo o una carpeta
        if os.path.isfile(ruta_completa) or os.path.islink(ruta_completa):
            os.remove(ruta_completa)  # Elimina archivos o enlaces simbólicos
        elif os.path.isdir(ruta_completa):
            shutil.rmtree(ruta_completa)  # Elimina subcarpetas y su contenido
            
def obtener_GBIF(path: str):
    return path.split('/')[-1]

def copiar_archivo(ruta_origen, ruta_destino):
    """Copia un archivo a una carpeta y si no existe la carpeta la crea"""
    if not os.path.exists(ruta_destino):
        os.makedirs(ruta_destino)
    shutil.copy(ruta_origen, f'{ruta_destino}/{obtener_GBIF(ruta_origen)}')
    
def convert_to_webp(input_image_path: str, output_image_path = "", quality=100, remove_original=True):
    """
    Convierte una imagen a formato WebP y la ajusta a un tamaño de imagen concreto.

    Args:
    input_image_path (str): La ruta de la imagen de entrada.
    output_image_path (str): La ruta donde se guardará la imagen convertida.
    width (int): El ancho deseado de la imagen.
    height (int): El alto deseado de la imagen.
    quality (int): La calidad de la imagen resultante, de 1 a 100.
    remove_original (bool): Si se debe eliminar la imagen original tras la conversión.
    """
    global imgsz
    if output_image_path == "":
        output_image_path = input_image_path.split(".")[0] + ".webp"
    try:
        with Image.open(input_image_path) as image:
            img_adjusted = image.resize((imgsz, imgsz), Image.LANCZOS)
            img_adjusted.save(output_image_path, 'WEBP', quality=quality)
            if remove_original:
                os.remove(input_image_path)

    except UnidentifiedImageError:
        print(f"Imagen no identificable: {input_image_path}, se procede a eliminar.")
        os.remove(input_image_path)

    except FileNotFoundError:
        print(f"Archivo no encontrado: {input_image_path}")

    except Exception as e:
        print(f"Se ha producido una excepción con la imagen: {input_image_path}\nError: {e}")
    return output_image_path

def es_imagen_corrupta(ruta_imagen):
    """
    Comprueba si una imagen está corrupta intentando abrirla con Pillow.

    Args:
    ruta_imagen (str): La ruta al archivo de imagen que se quiere comprobar.

    Returns:
    bool: True si la imagen está corrupta o no puede ser identificada; False en caso contrario.
    """
    try:
        with Image.open(ruta_imagen) as img:
            img.verify()  # Intenta verificar la imagen
        return False  # La imagen se abrió y verificó correctamente
    except UnidentifiedImageError:
        return True  # La imagen está corrupta o no es un formato identificable
    except Exception:
        return False
    

def transformar_imagen_webp(ruta: str, numero: int):
    """Transforma una imagen de tipo webp para así tener mas datos de entrenamiento"""
    # Cargar una imagen
    if ".webp" in ruta:
        imagen = imageio.imread(ruta)
        # Obtener dimensiones


        # Definir una secuencia de aumentos
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # aplicar espejado horizontal con una probabilidad del 50%
            iaa.Affine(rotate=(-25, 25)),  # rotar la imagen entre -25 y 25 grados
            iaa.Multiply((0.8, 1.2)),  # cambiar el brillo de la imagen (80-120% del original)
        ])

        # Aplicar los aumentos a la imagen
        imagen_augmentada = seq.augment_image(imagen)

        ruta_salida = ruta.replace(".webp",f"_{numero}.webp")
        # Guardar la imagen aumentada
        imageio.imwrite(ruta_salida, imagen_augmentada)

def descartar_imagen_mala(img_path, preguntar=False,confianza=0.90):
    """Descarta las imagenes que no nos son utiles para el proyecto segun el modelo ya entrenado por YOLo
    
    Args:
    img_path: el path a la imagen a predecir
    preguntar: si quieres de que en caso de que sea menor del 85% de confianza te pregunte, por defecto False
    confianza: la confianza que quieres que tenga el modelo entre(0,1) por defecto 0.85
    
    return: devuelve True si es buena, False si es mala o bien no es un fichero"""
    # Realiza la predicción en la imagen
    results = model(img_path)
    
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

        if (result.names[top1_class_index] == tipos['mala']):
            if os.path.isfile(img_path) or os.path.islink(img_path):
                
                if (top1_confidence < confianza and preguntar):
                    image = Image.open(img_path)
                    # Mostrar la imagen
                    image.show()
                    print(f"Clase más probable: {result.names[top1_class_index]} con confianza {top1_confidence}\n")
                    respuesta = input('¿Desea eliminar la imagen? (s/n)\n')
                    if (respuesta == 's'):
                        os.remove(img_path) 
                        return False # La imagen es mala
                    else: return True # La imagen es buena
                else: 
                    os.remove(img_path)
                    return False # La imagen es mala
            else: return False # La imagen no es un fichero
        else: return True # La imagen es buena

def shuffleDataFrame(df: pd.DataFrame):
    """Esta función desordena un dataframe para que sea aleatorio y guarda el resultado en un nuevo archivo CSV.

    Args:
        strFile (str): Nombre del fichero csv que hay que desordenar.
    """
    
    # Se define el nombre del archivo de salida agregando 'shuffled_' al inicio del nombre de archivo original.
    
    # Inicializar una lista vacía para almacenar los chunks desordenados.
    chunks_list = []
    
    # Desordenar cada chunk y agregarlo a la lista de chunks.
    for chunk in df:
        shuffled_chunk = chunk.sample(frac=1).reset_index(drop=True)
        chunks_list.append(shuffled_chunk)
    
    # Concatenar todos los chunks desordenados en un solo DataFrame.
    shuffled_df = pd.concat(chunks_list, ignore_index=True)
    
    shuffled_df.to_csv(csv_datos, index=False)
    # Guardar el DataFrame desordenado en un nuevo archivo CSV.
    print(f"Archivo desordenado guardado como: {csv_datos}")
    
    return chunks_list
