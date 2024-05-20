"""Este fichero de python se encarga de procesar el fichero de pandas para eliminar aquellas filas donde la imagen no sea valida segun el modelo de ia.
"""

from globales import *
from ultralytics import YOLO
from tqdm import tqdm
import os
import pandas as pd
import requests
from requests.exceptions import RequestException
from random import  randint
from functools import reduce
from funciones import *
rangos_taxonomicos = [
    'class',
    'order',
    'family',
    'genus',
    'acceptedScientificName'
]

def procesar_pandas():   
    
    chunksize = 10**4
    vaciar_carpeta(RUTA_IMG_TEMPORALES) # Vaciamos la carpeta de imagenes de entrenamiento.
    
    df_occurrences = pd.read_csv(CSV_DATOS_TODAS, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
    
    filas = []
    
    for chunk in df_occurrences:
        for indice_fila, fila in tqdm(pd.DataFrame(chunk).iterrows(), desc="Procesando elementos", unit="elementos"):
                    
            # Comprobar si la fila tiene un identificador válido
            if pd.notna(fila['identifier']):

                # Construir la ruta de la carpeta basada en la clasificación taxonómica
                ruta_carpeta = os.path.join(RUTA_IMG_TEMPORALES,'prueba')

                # Crear la carpeta si no existe
                if not os.path.exists(ruta_carpeta):
                    os.makedirs(ruta_carpeta)

                # Construir la ruta completa del archivo de imagen a guardar
                ruta_completa = os.path.join(ruta_carpeta, parsear_nombre(str(fila['gbifID'])) + ".jpg")
                ruta_webp = ruta_completa.replace(".jpg",".webp")
                
                # Descargar y guardar la imagen si aún no existe
                if ((not os.path.exists(ruta_completa) or es_imagen_corrupta(ruta_completa)) and not os.path.exists(ruta_webp)):
                    try:
                        respuesta = requests.get(fila['identifier'],timeout=5)
                        if respuesta.status_code == 200:
                            with open(ruta_completa, 'wb') as archivo:
                                archivo.write(respuesta.content)
                                archivo.close()
                                if (descartar_imagen_mala(ruta_completa)):
                                    filas.append(fila)
                                if os.path.exists(ruta_completa):
                                    os.remove(ruta_completa)
                        
                    
                    except KeyboardInterrupt:
                        # guardar_ficheros()
                        info("El usuario ha detenido el programa")
                        exit(-1)

                    except RequestException as e:
                        warning("Fallo en la URL : " + fila['identifier'] + str(e))
                        # Agregar la fila al DataFrame de registros fallidos si ocurre un error
                        # df_fallidos = pd.concat([df_fallidos, pd.DataFrame([fila])], ignore_index=True)
                        
                    except Exception as e: 
                        fail(f"Ha ocurrido un error, {e}")
                        pass
                        # Agregar la fila al DataFrame de registros fallidos si ocurre un error
                        # df_fallidos = pd.concat([df_fallidos, pd.DataFrame([fila])], ignore_index=True)
        else:
            pass
                     
    df_filas_validas = pd.DataFrame(filas)
    # Guardar el DataFrame en un archivo CSV
    df_filas_validas.to_csv(CSV_DATOS, index=False)

if __name__ == "__main__":
    procesar_pandas()