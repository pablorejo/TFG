import globales as gb
from ultralytics import YOLO
from tqdm import tqdm
import os
import pandas as pd
import requests
from requests.exceptions import RequestException
from random import random, randint

def entrenar_carpeta(nombre_modelo:str, nombre_carpeta:str):
    """Entrena los datos de una carpeta y sus subcarpetas.
    
    Args:
    nombre_modelo: el nombre con el que se guardará el modelo
    nombre_carpeta: la direccion de la carpeta a la que se va a entrenar"""
    # Entrenar dependiendo todo de golpe
    directorios = gb.obtener_directorios(nombre_carpeta)

    for clase in directorios:
        imagenes = gb.encontrar_imagenes_jpg(f'{nombre_carpeta}/{clase}')
        gb.copiar_a_training(tipo=clase,lineas=imagenes)

    model = YOLO('yolov8n-cls.pt') 
    results = model.train(data='training.yaml', epochs=gb.epocas_entrenamiento, imgsz=gb.imgsz, name=nombre_modelo)

entrenar_local = False #Esto quiere decir que se va a entrenar de manera local con las imagenes que están ya descargadas en la carpeta definida glovales.carpeta_de_imagenes

if (entrenar_local):
    for ruta in gb.ruta_training_data.values():
        gb.vaciar_carpeta(ruta)

    directorios = gb.obtener_directorios(gb.carpeta_de_imagenes,gb.carpeta_de_imagenes)
    entrenar_carpeta(gb.rangos_taxonomicos[0][0],gb.carpeta_de_imagenes)

    for taxon,indice in gb.rangos_taxonomicos:
        for nombre in gb.obtener_carpetas_nivel(gb.carpeta_de_imagenes,nivel_max=indice):
            directorios = gb.obtener_directorios(gb.carpeta_de_imagenes,gb.carpeta_de_imagenes)
            # nombre_modelo = (f"{taxon}_{nombre.split("/")[-1]}")
            # entrenar_carpeta(nombre_modelo,nombre)
            
else:
    diferentes_entrenamientos = [
        "class",
        "order",
        "family",
        "genus",
        "acceptedScientificName"
    ]

    for entrenamiento in diferentes_entrenamientos:
        chunksize = 10**4
        gb.vaciar_carpeta(gb.ruta_destino_training) # Vaciamos la carpeta de imagenes de entrenamiento.
        
        df_occurrences = pd.read_csv(gb.csv_datos, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
        conteos_totales = {} #Contiene un diccionario de el nombre del toponimo con el numero de datos que existen de este.
        
        conteos_iniciales = {}
        for chunk in df_occurrences:
            values = []
            keys = []
            
            for valor in chunk[entrenamiento].value_counts().values:
                values.append(valor)
            for clave in chunk[entrenamiento].value_counts().keys().values:
                keys.append(clave)
            
            for i in range(len(values)):
                
                if keys[i] not in conteos_totales:
                    conteos_totales[keys[i]] = 0
                    conteos_iniciales[keys[i]] = 0
                conteos_totales[keys[i]] += values[i]
                
        
        df_occurrences = pd.read_csv(gb.csv_datos, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
        ruta_anterior = ""
        
        if(gb.DESORDENAR_DATAFRAME):
            df_occurrences = gb.shuffleDataFrame(df_occurrences)
        
        for chunk in df_occurrences:
            for indice, fila in tqdm(pd.DataFrame(chunk).iterrows(), desc="Procesando elementos", unit="elementos"):
                if fila[entrenamiento] not in conteos_iniciales:
                    conteos_iniciales[fila[entrenamiento]] = 0
                    
                if conteos_iniciales[fila[entrenamiento]] <= gb.numero_de_muestras_imagen: #Si el taxon ya tiene todas las imagenes necesarias no descarga mas.
                
                    # Comprobar si la fila tiene un identificador válido
                    if pd.notna(fila['identifier']):

                        # Construir la ruta de la carpeta basada en la clasificación taxonómica
                        ruta_carpeta = f"{gb.ruta_destino_training}/{gb.parsear_nombre(fila[entrenamiento])}"

                        # Crear la carpeta si no existe
                        if not os.path.exists(ruta_carpeta):
                            os.makedirs(ruta_carpeta)

                        # Construir la ruta completa del archivo de imagen a guardar
                        ruta_completa = os.path.join(ruta_carpeta, gb.parsear_nombre(str(fila['gbifID'])) + ".jpg")
                        ruta_webp = ruta_completa.replace(".jpg",".webp")
                        
                        # Descargar y guardar la imagen si aún no existe
                        if ((not os.path.exists(ruta_completa) or gb.es_imagen_corrupta(ruta_completa)) and not os.path.exists(ruta_webp)):
                            try:
                                    
                                    respuesta = requests.get(fila['identifier'],timeout=5)
                                    if respuesta.status_code == 200:
                                        with open(ruta_completa, 'wb') as archivo:
                                            archivo.write(respuesta.content)
                                            if (gb.descartar_imagen_mala(ruta_completa)):
                                                ruta_anterior = gb.convert_to_webp(ruta_completa)
                                                conteos_iniciales[fila[entrenamiento]] += 1
                                
                                    # Guardamos los que si que se han completado con exito.
                                    # df_completados = pd.concat([df_completados, pd.DataFrame([fila])], ignore_index=True)
                            
                            except KeyboardInterrupt:
                                # guardar_ficheros()
                                print("El programa ha finalizado con falllos con éxito")
                                exit(-1)

                            except RequestException as e:
                                print("Fallo en la URL : " + fila['identifier'])
                                print(e)
                                # Agregar la fila al DataFrame de registros fallidos si ocurre un error
                                # df_fallidos = pd.concat([df_fallidos, pd.DataFrame([fila])], ignore_index=True)
                                
                            except Exception as e: 
                                pass
                                # Agregar la fila al DataFrame de registros fallidos si ocurre un error
                                # df_fallidos = pd.concat([df_fallidos, pd.DataFrame([fila])], ignore_index=True)
                else:
                    print("Termino el taxon: "+entrenamiento)        
        # aumentar imagenes
        for key, value in conteos_iniciales.items():
            if (value < gb.numero_de_muestras_imagen):
                ruta_carpeta = f"{gb.ruta_destino_training}/{key}/"
                imagenes = gb.encontrar_imagenes_jpg(ruta_carpeta)
                i = 0
                while value < gb.numero_de_muestras_imagen:
                    gb.transformar_imagen_webp(ruta_carpeta+imagenes[randint(0,len(imagenes)-1)],i)
                    i += 1
                    value += 1
                
        
        model = YOLO('yolov8n-cls.pt') 
        nombre_modelo = "modelo_entrenando_"+ entrenamiento + ".pt"
        carpeta_modelo = 'runs/classify/'+ nombre_modelo
        
            
        results = model.train(epochs=gb.epocas_entrenamiento, imgsz=gb.imgsz, name=nombre_modelo)