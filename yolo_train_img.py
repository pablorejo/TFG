import globales
from ultralytics import YOLO

def entrenar_carpeta(nombre_modelo:str,nombre_carpeta:str):
    """Entrena los datos de una carpeta y sus subcarpetas.
    
    Args:
    nombre_modelo: el nombre con el que se guardará el modelo
    nombre_carpeta: la direccion de la carpeta a la que se va a entrenar"""
    # Entrenar dependiendo todo de golpe
    directorios = globales.obtener_directorios(nombre_carpeta)

    for clase in directorios:
        imagenes = globales.encontrar_imagenes_jpg(f'{nombre_carpeta}/{clase}')
        globales.copiar_a_training(tipo=clase,lineas=imagenes)

    model = YOLO('yolov8n-cls.pt') 
    results = model.train(data='training.yaml', epochs=1, imgsz=256, name=nombre_modelo)


for ruta in globales.ruta_training_data.values():
    globales.vaciar_carpeta(ruta)

# for categori_taxonomica in globales.rangos_taxonomicos:
    


directorios = globales.obtener_directorios(globales.carpeta_de_imagenes,globales.carpeta_de_imagenes)
entrenar_carpeta(globales.rangos_taxonomicos[0][0],globales.carpeta_de_imagenes)

for taxon,indice in globales.rangos_taxonomicos:
    for nombre in globales.obtener_carpetas_nivel(globales.carpeta_de_imagenes,nivel_max=indice):
        directorios = globales.obtener_directorios(globales.carpeta_de_imagenes,globales.carpeta_de_imagenes)
        entrenar_carpeta(f"{taxon}_{nombre.split("/")[-1]}",nombre)
