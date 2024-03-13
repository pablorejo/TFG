import os, platform, random
import shutil #Para copiar las imagenes en la carpeta de entrenamiento
import random,math

carpeta_de_imagenes = 'imagenes'

nombre_archivo_buenas = 'imagenes_buenas.txt'
nombre_archivo_malas = 'imagenes_malas.txt'


# Datos de entrenamiento
ruta_destino = 'training/'

ruta = {
    'train': ruta_destino+'train',
    'test': ruta_destino+'test',
    'valid': ruta_destino+'valid',
}

tipos = {
    'buena': 'buenas',
    'mala': 'malas'
}

rangos_taxonomicos = [
    'clase',
    'orden',
    'familia',
    'genero',
    'especie'
]

porcentaje_validacion = 0.1
porcentaje_testing = 0.02
porcentaje_training = 1 - porcentaje_testing - porcentaje_validacion

def copiar_a_training(tipo, file:str):
    with open(file) as file:
        lineas = file.read().splitlines()
        random.shuffle(lineas)  # Esto modifica la lista "in-place"
        copiar_a_training(tipo,lineas=lineas)

def copiar_a_training(tipo:str, lineas):
    num_valid = math.ceil(len(lineas) * porcentaje_validacion)
    num_test = math.ceil(len(lineas) * porcentaje_testing)
    
    for i in range(num_valid):
        copiar_archivo(lineas[i], f'{ruta["valid"]}/{tipo}')
    
    for i in range(num_valid,num_valid + num_test):
        copiar_archivo(lineas[i], f'{ruta["test"]}/{tipo}')
        
    for i in range(num_valid + num_test, len(lineas)):
        copiar_archivo(lineas[i], f'{ruta["train"]}/{tipo}')

            
def encontrar_imagenes_jpg(directorio):
    imagenes_jpg = []
    for root, dirs, files in os.walk(directorio):
        for file in files:
            if file.lower().endswith('.jpg'):
                imagenes_jpg.append(os.path.join(root, file))
    return imagenes_jpg

def obtener_directorios(ruta_carpeta):
    """Obtiene una lista de los nombres de los directorios (no archivos) dentro de la ruta_carpeta dada."""
    directorios = [nombre for nombre in os.listdir(ruta_carpeta)
    if os.path.isdir(os.path.join(ruta_carpeta, nombre))]
    
    return directorios

def encontrar_imagenes_jpg(directorio, num_muestras=1000):
    imagenes_jpg = []
    n = 0  # Contador total de archivos jpg encontrados
    
    for root, dirs, files in os.walk(directorio):
        for file in files:
            if file.lower().endswith('.jpg'):
                n += 1
                # Si aún no hemos recolectado num_muestras imágenes, simplemente añádelas
                if len(imagenes_jpg) < num_muestras:
                    imagenes_jpg.append(os.path.join(root, file))
                else:
                    # Con probabilidad num_muestras/n, reemplaza un elemento aleatorio
                    s = int(random.random() * n)
                    if s < num_muestras:
                        imagenes_jpg[s] = os.path.join(root, file)
    
    return imagenes_jpg

def apagar_equipo():
    if platform.system() == "Windows":
        os.system('shutdown /s /t 1')
    elif platform.system() == "Linux":
        os.system('shutdown -h now')
    elif platform.system() == "Darwin":
        os.system('shutdown -h now')  
        
def vaciar_carpeta(ruta_carpeta):
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
    if not os.path.exists(ruta_destino):
        os.makedirs(ruta_destino)
    shutil.copy(ruta_origen, f'{ruta_destino}/{obtener_GBIF(ruta_origen)}')
    
