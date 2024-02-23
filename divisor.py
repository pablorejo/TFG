def obtener_primeras_3000_lineas(archivo_entrada, archivo_salida):
    """
    Lee las primeras 3000 líneas de un archivo y las escribe en otro archivo.

    :param archivo_entrada: La ruta al archivo de entrada.
    :param archivo_salida: La ruta al archivo de salida donde se guardarán las 3000 primeras líneas.
    """
    try:
        with open(archivo_entrada, 'r', encoding='utf-8') as fuente:
            with open(archivo_salida, 'w', encoding='utf-8') as destino:
                for i, linea in enumerate(fuente):
                    if i < 30:
                        destino.write(linea)
                    else:
                        break
    except FileNotFoundError:
        print(f"No se pudo encontrar el archivo: {archivo_entrada}")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Usar la función
archivo_entrada = 'ocurrencias_parseado.csv'
archivo_salida = 'ocurrencias_2.csv'
# archivo_entrada = 'multimedia.txt'
# archivo_salida = 'multimedia_2.txt'
obtener_primeras_3000_lineas(archivo_entrada, archivo_salida)


