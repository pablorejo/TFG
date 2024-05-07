from globales import *
from ultralytics import YOLO
from tqdm import tqdm
import os
import pandas as pd
import requests
from requests.exceptions import RequestException
from random import  randint
from functools import reduce

rangos_taxonomicos = [
    'class',
    'order',
    'family',
    'genus',
    'acceptedScientificName'
]

def entrenar_carpeta(nombre_modelo:str, nombre_carpeta:str):
    """Entrena los datos de una carpeta y sus subcarpetas.
    Lo hace de tal forma que 
    
    Args:
    nombre_modelo: el nombre con el que se guardará el modelo
    nombre_carpeta: la direccion de la carpeta a la que se va a entrenar"""
    # Entrenar dependiendo todo de golpe
    directorios = obtener_directorios(nombre_carpeta)

    for clase in directorios:
        imagenes = encontrar_imagenes(f'{nombre_carpeta}/{clase}')
        copiar_a_training(tipo=clase,lineas=imagenes)

    model = YOLO('yolov8n-cls.pt') 
    model.val(imgsz=IMGSZ)
    results = model.train(data='training.yaml', epochs=EPOCAS_DE_ENTRENAMIENTO, imgsz=IMGSZ, name=nombre_modelo)

def entrenar(
            str_nombre_modelo_inicio: str = "modelo",
            model = YOLO('yolov8n-cls.pt'),
            filtros_columna : list  = [],
            filtros: list = [],
            indice_taxon: int = 0):   
    
    chunksize = 10**4
    vaciar_carpeta(RUTA_IMG_TEMPORALES) # Vaciamos la carpeta de imagenes de entrenamiento.
    entrenamiento = rangos_taxonomicos[indice_taxon]
    
    df_occurrences = pd.read_csv(CSV_DATOS, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
    
    conteos_totales = {} #Contiene un diccionario de el nombre del toponimo con el numero de datos que existen de este.
    
    conteos_iniciales = {}
    dfs = []
    for chunk in df_occurrences:
        if entrenamiento and len(filtros_columna) == len(filtros):
            # Aplicar cada filtro de columna especificado
            for col_name, filter_value in zip(filtros_columna, filtros):
                chunk = chunk[chunk[col_name] == filter_value]
                
        # if (entrenamiento != None):
        #     for k in range(0,len(filtros_columna)-1):
        #         filtros_aplicados = [chunk[filtros_columna[i]] == filtros[i] for i in range(len(filtros_columna))]
        #         chunk = chunk[reduce(lambda x, y: x & y, filtros_aplicados)]
        
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
            
        dfs.append(chunk)# Añadimos el chunk 
            
    
    
    if(DESORDENAR_DATAFRAME):
        dfs = shuffleDataFrame(dfs)
    
    for chunk in dfs:
        for indice_fila, fila in tqdm(pd.DataFrame(chunk).iterrows(), desc="Procesando elementos", unit="elementos"):
            seguir = True
            
            # Como ya est
            # for k in range(0,len(filtros_columna)-1):
            #     if (fila[filtros_columna[k]] != filtros[k]):
            #         seguir = False
            #         break
                
            if seguir:
            
                if fila[entrenamiento] not in conteos_iniciales:
                    conteos_iniciales[fila[entrenamiento]] = 0
                    
                if conteos_iniciales[fila[entrenamiento]] < NUMERO_DE_MUESTRAS_IMAGEN: #Si el taxon ya tiene todas las imagenes necesarias no descarga mas.
                
                    # Comprobar si la fila tiene un identificador válido
                    if pd.notna(fila['identifier']):

                        # Construir la ruta de la carpeta basada en la clasificación taxonómica
                        ruta_carpeta = os.path.join(RUTA_IMG_TEMPORALES,parsear_nombre(fila[entrenamiento]))

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
                                                rutas = recortarImagenes(src_img=ruta_completa) 
                                                for ruta in rutas:
                                                    convert_to_webp(ruta)
                                                conteos_iniciales[fila[entrenamiento]] += len(rutas)
                                
                            
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
                    pass
                        
    # aumentar imagenes
    for key, value in conteos_iniciales.items():
        if (value < NUMERO_DE_MUESTRAS_IMAGEN):
            ruta_carpeta = f"{RUTA_IMG_TEMPORALES}/{key}/"
            imagenes = encontrar_imagenes(ruta_carpeta)
            if (len(imagenes) > 0):
                while value < NUMERO_DE_MUESTRAS_IMAGEN:
                    transformar_imagen_webp(imagenes[randint(0,len(imagenes)-1)],value)
                    value += 1
            else:
                print(f"{Colors.WARNING}No hay imagenes de la clase: {key}{Colors.ENDC}")
    
    
    if (len(filtros_columna) == 0):
        nombre_modelo = str_nombre_modelo_inicio 
    else: 
        nombre_modelo = str_nombre_modelo_inicio + "_" + filtros_columna[-1] +"_" + filtros[-1] 
    
    carpeta_modelo = 'runs/classify/'+ nombre_modelo
    
    if (os.path.exists(carpeta_modelo)):
        vaciar_carpeta(carpeta_modelo)
        os.rmdir(carpeta_modelo)
        
    copiar_a_training(RUTA_IMG_TEMPORALES)
    print("Entrenando: " + nombre_modelo)
    results = model.train(epochs=EPOCAS_DE_ENTRENAMIENTO, imgsz=IMGSZ, name=nombre_modelo)
    model = YOLO(os.path.join(carpeta_modelo,'weights','best.pt'),task='')

    for key, value in conteos_totales.items():
        if (indice_taxon < len(rangos_taxonomicos)-1):
            filtros_columna.append(rangos_taxonomicos[indice_taxon])
            filtros.append(key)
            entrenar(filtros_columna=filtros_columna, filtros=filtros,indice_taxon=indice_taxon+1, model=model)
        else:
            print(f"{Colors.OKGREEN}Termino el {filtros_columna} de {filtros}")

if __name__ == "__main__":
    if (ENTRENAR_LOCAL):
        for ruta in ruta_training_data.values():
            vaciar_carpeta(ruta)

        directorios = obtener_directorios(CARPETA_IMAGENES)
        entrenar_carpeta(rangos_taxonomicos[0][0],CARPETA_IMAGENES)

        for taxon,indice in rangos_taxonomicos:
            for nombre in obtener_carpetas_nivel(CARPETA_IMAGENES,nivel_max=indice):
                directorios = obtener_directorios(CARPETA_IMAGENES)
                nombre = str(nombre).split("/")[-1]
                nombre_modelo = (f"{taxon}_{nombre}")
                entrenar_carpeta(nombre_modelo,nombre)
    else:
        entrenar()