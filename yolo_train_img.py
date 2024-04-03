import globales as gb
from ultralytics import YOLO
from tqdm import tqdm
import os
import pandas as pd
import requests
from requests.exceptions import RequestException
from random import random, randint

rangos_taxonomicos = [
    'class',
    'order',
    'family',
    'genus'
    # ,('especie',5)
]

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

def entrenar(
            str_nombre_modelo_inicio: str = "modelo",
            model = YOLO('yolov8n-cls.pt'),
            filtro_columna : str  = "class",
            filtro: str = None,
            indice_taxon: int = 0):   
    
    chunksize = 10**4
    gb.vaciar_carpeta(gb.ruta_destino_training) # Vaciamos la carpeta de imagenes de entrenamiento.
    
    df_occurrences = pd.read_csv(gb.csv_datos, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
    
    conteos_totales = {} #Contiene un diccionario de el nombre del toponimo con el numero de datos que existen de este.
    
    conteos_iniciales = {}
    dfs = []
    for chunk in df_occurrences:
        if (filtro_columna != None and filtro != None):
            chunk = chunk.loc[chunk[filtro_columna] == filtro] # Filtramos el chunk en caso de ser necesario.
        
        values = []
        keys = []
        
        for valor in chunk[filtro_columna].value_counts().values:
            values.append(valor)
        for clave in chunk[filtro_columna].value_counts().keys().values:
            keys.append(clave)
        
        for i in range(len(values)):
            if keys[i] not in conteos_totales:
                conteos_totales[keys[i]] = 0
                conteos_iniciales[keys[i]] = 0
            conteos_totales[keys[i]] += values[i]
            
        dfs.append(chunk)# Añadimos el chunk 
            
    
    
    if(gb.DESORDENAR_DATAFRAME):
        dfs = gb.shuffleDataFrame(dfs)
    
    for chunk in dfs:
        for indice_fila, fila in tqdm(pd.DataFrame(chunk).iterrows(), desc="Procesando elementos", unit="elementos"):
          
            if ((filtro_columna == None or filtro == None or fila[filtro_columna] == filtro )):
            
                if fila[filtro_columna] not in conteos_iniciales:
                    conteos_iniciales[fila[filtro_columna]] = 0
                    
                if conteos_iniciales[fila[filtro_columna]] <= gb.numero_de_muestras_imagen: #Si el taxon ya tiene todas las imagenes necesarias no descarga mas.
                
                    # Comprobar si la fila tiene un identificador válido
                    if pd.notna(fila['identifier']):

                        # Construir la ruta de la carpeta basada en la clasificación taxonómica
                        ruta_carpeta = f"{gb.ruta_destino_training}/{gb.parsear_nombre(fila[filtro_columna])}"

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
                                                conteos_iniciales[fila[filtro_columna]] += 1
                                
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
                    pass
                        
    # aumentar imagenes
    for key, value in conteos_iniciales.items():
        if (value < gb.numero_de_muestras_imagen):
            ruta_carpeta = f"{gb.ruta_destino_training}/{key}/"
            imagenes = gb.encontrar_imagenes_jpg(ruta_carpeta)
            while value < gb.numero_de_muestras_imagen:
                gb.transformar_imagen_webp(imagenes[randint(0,len(imagenes)-1)],value)
                value += 1
    
    
    if (filtro_columna == None or filtro == None):
        nombre_modelo = str_nombre_modelo_inicio + ".pt"
    else: 
        nombre_modelo = str_nombre_modelo_inicio + "_" + filtro_columna +"_" + filtro + ".pt"
        
    carpeta_modelo = 'runs/classify/'+ nombre_modelo
    
    if (not os.path.exists(carpeta_modelo)):
        gb.vaciar_carpeta(carpeta_modelo)
        os.rmdir(carpeta_modelo)
        
    results = model.train(epochs=gb.epocas_entrenamiento, imgsz=gb.imgsz, name=nombre_modelo)
    
    model = YOLO(nombre_modelo)
    
    for key, value in conteos_totales.items():
        if (indice_taxon < len(rangos_taxonomicos)-1):
            entrenar(filtro_columna=rangos_taxonomicos[indice_taxon], filtro=key,indice_taxon=indice_taxon+1, model=model)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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
    
    entrenar()