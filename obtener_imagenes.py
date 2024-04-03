import pandas as pd
from tqdm import tqdm
import math, os
import requests
from requests.exceptions import RequestException
import globales as gb

def guardar_ficheros():    
    # Guardar el DataFrame de registros fallidos en un archivo CSV
    nombre_fallidos = 'ocurrencias_fallidas.csv'
    df_fallidos.to_csv(nombre_fallidos, index=False)

    # Guardar el DataFrame de registros completados en un archivo CSV
    nombre_completados = 'ocurrencias_completados.csv'
    df_completados.to_csv(nombre_completados, index=False)

if __name__ == '__main__':
    # Comprobamos que estamos como superusuario
    if os.geteuid() != 0:
        print("El programa necesita ser ejecutado como superusuario\nsudo su")
        exit(-1)
        
    # Definir la ruta base para guardar las imágenes descargadas
    RUTA_IMAGENES = 'imagenes'
    # Configurar Pandas para mostrar la totalidad de las columnas de texto
    pd.set_option('display.max_colwidth', None)
    # Ruta del archivo CSV que contiene las ocurrencias
    ruta_del_archivo_occurrences = 'ocurrencias_parseado.csv'

    # Tamaño de cada "chunk" al leer el archivo CSV: 1 millón de filas por vez
    chunksize = 1 * 10 ** 6 

    print("Leyendo el fichero de ocurrencias")
    # Leer el archivo CSV en fragmentos para manejar grandes volúmenes de datos
    df_occurrences = pd.read_csv(ruta_del_archivo_occurrences, delimiter=',', low_memory=False, on_bad_lines='skip')

    # Calcular el tamaño total del archivo para estimar el número de chunks
    espacio = os.path.getsize(ruta_del_archivo_occurrences)
    # Calcular el número total de chunks basado en el tamaño del archivo y el tamaño de chunk
    total_chunks = math.ceil((espacio/166)/chunksize)

    print("Iniciando chunks\n")

    # Procesar solo el primer chunk para inicializar el DataFrame de registros fallidos
    df_fallidos = pd.DataFrame(df_occurrences.columns)
    df_completados = pd.DataFrame(df_occurrences.columns)

    print(df_occurrences.head())
    taxones = df_occurrences[df_occurrences['class'] == 'Bivalvia']

    try:
        # Inicializar la barra de progreso para el procesamiento total de elementos
        with tqdm(total=total_chunks, desc="Procesando elementos", unit="elemento") as pbar_total:
            es_primer_chunk = True  # Indicador para saber si estamos procesando el primer chunk
            nombre_archivo = 'ocurrencias.csv'

            # Iterar sobre cada fila del DataFrame actual
            for indice, fila in tqdm(df_occurrences.iterrows(), desc="Procesando elementos", unit="elementos"):

                # Comprobar si la fila tiene un identificador válido
                if pd.notna(fila['identifier']):

                    # Construir la ruta de la carpeta basada en la clasificación taxonómica
                    ruta_carpeta = f"{RUTA_IMAGENES}/{gb.parsear_nombre(fila['class'])}/{gb.parsear_nombre(fila['order'])}/{gb.parsear_nombre(fila['family'])}/{gb.parsear_nombre(fila['genus'])}/{gb.parsear_nombre(fila['acceptedScientificName'])}"

                    # Crear la carpeta si no existe
                    if not os.path.exists(ruta_carpeta):
                        os.makedirs(ruta_carpeta)

                    # Construir la ruta completa del archivo de imagen a guardar
                    ruta_completa = os.path.join(ruta_carpeta, gb.parsear_nombre(str(fila['gbifID'])) + ".jpg")
                    ruta_webp = ruta_completa.replace(".jpg",".webp")


                    
                    # Descargar y guardar la imagen si aún no existe
                    if (not os.path.exists(ruta_completa) or gb.es_imagen_corrupta(ruta_completa) and not os.path.exists(ruta_webp)):
                        try:
                            os.remove(ruta_completa)
                            respuesta = requests.get(fila['identifier'],timeout=5)
                            if respuesta.status_code == 200:
                                with open(ruta_completa, 'wb') as archivo:
                                    archivo.write(respuesta.content)
                                    gb.convert_to_webp(ruta_completa)
                                # Guardamos los que si que se han completado con exito.
                                df_completados = pd.concat([df_completados, pd.DataFrame([fila])], ignore_index=True)
                        
                        except KeyboardInterrupt:
                            guardar_ficheros()
                            print("El programa ha finalizado con falllos con éxito")
                            exit(-1)

                        except RequestException as e:
                            print("Fallo en la URL : " + fila['identifier'])
                            print(e)
                            # Agregar la fila al DataFrame de registros fallidos si ocurre un error
                            df_fallidos = pd.concat([df_fallidos, pd.DataFrame([fila])], ignore_index=True)
                            
                        except Exception as e: 
                            
                            # Agregar la fila al DataFrame de registros fallidos si ocurre un error
                            df_fallidos = pd.concat([df_fallidos, pd.DataFrame([fila])], ignore_index=True)

            # Actualizar la barra de progreso después de procesar cada chunk
            pbar_total.update(1)

        guardar_ficheros()
        print("El programa ha finalizado con éxito")
        # gb.apagar_equipo()

    except Exception as e:
        guardar_ficheros()
        print("El programa ha finalizado con falllos con éxito")
        print(f"Excepcion {e}")
        # gb.apagar_equipo()
