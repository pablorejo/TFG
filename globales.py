import os, platform, random
import shutil #Para copiar las imagenes en la carpeta de entrenamiento
import random,math
from PIL import Image,UnidentifiedImageError
from imgaug import augmenters as iaa
import imageio
import numpy as np

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
ruta_destino_training = 'training/'
ruta_training_data = {
    'train': ruta_destino_training+'train',
    'test': ruta_destino_training+'test',
    'valid': ruta_destino_training+'valid',
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
    
def convert_to_webp(input_image_path, output_image_path, quality=90):
    """
    Convierte una imagen a formato WebP.

    Args:
    input_image_path (str): La ruta de la imagen de entrada.
    output_image_path (str): La ruta donde se guardará la imagen convertida.
    quality (int): La calidad de la imagen resultante, de 1 a 100.
    """

    try:
        image = Image.open(input_image_path)

        # Convertir y guardar la imagen en formato WebP
        image.save(output_image_path, 'WEBP', quality=quality)
        os.remove(input_image_path)

    except UnidentifiedImageError:
        try:
            # Si se lanza UnidentifiedImageError, elimina el archivo de imagen no identificado
            os.remove(input_image_path)
        except FileNotFoundError:
            pass
    except FileNotFoundError:
            pass

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
    

def transformar_imagen_webp(ruta: str, numero: int):
    """Transforma una imagen de tipo webp para así tener mas datos de entrenamiento"""
    # Cargar una imagen
    if ".webp" in ruta:
        imagen = imageio.imread(ruta)

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
