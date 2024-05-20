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


def descargar_imagenes_aleatorio(n_imagenes_descargar = 1000, todas = False):  
    """
    Esta funcion lo que hace es descargar un numero predeterminado de imagenes de un archivo de pandas de forma aleatoria
    Args:
    n_imagenes_descargar = Es el numero de imagenes que se van a descargar
    todas = boleano que indica si se tienen que descargar todas las imágenes o no, esto quiere decir que si se tienen que descartar las imágenes malas o no""" 
    
    chunksize = 10**4
    df_occurrences = pd.read_csv(CSV_DATOS, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
    
    if(DESORDENAR_DATAFRAME):
        df_occurrences = shuffle_DataFrame(df_occurrences)
    n_imagenes = 0
    
    CARPETA_IMAGENES_ENTRENAR_DETECCION = 'imagenes_deteccion'
    
    vaciar_carpeta(RUTA_IMG_DETECCION)
    
    for chunk in df_occurrences:
        if (n_imagenes == n_imagenes_descargar):
            break
        else:
            for indice_fila, fila in tqdm(pd.DataFrame(chunk).iterrows(), desc="Procesando elementos", unit="elementos"):
                if (n_imagenes < n_imagenes_descargar):
                    # Comprobar si la fila tiene un identificador válido
                    if pd.notna(fila['identifier']):

                        # Construir la ruta de la carpeta basada en la clasificación taxonómica
                        ruta_carpeta = RUTA_IMG_DETECCION

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
                                                    
                                            if not todas:
                                                if (descartar_imagen_mala(ruta_completa)):
                                                    if model_detect != None:
                                                        recortar_imagenes(ruta_completa,delete_original=False)
                                                    ruta_anterior = convert_to_webp(ruta_completa,only_rescal=True)
                                                    n_imagenes += 1
                                            else:
                                                ruta_anterior = convert_to_webp(ruta_completa,only_rescal=True)
                                                n_imagenes += 1
                                            
                                            
                                                
                            
                            except KeyboardInterrupt:
                                # guardar_ficheros()
                                print("El programa ha finalizado con fallos con éxito")
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
                    break
                        
if __name__ == "__main__":
    descargar_imagenes_aleatorio(2000,True)
    

