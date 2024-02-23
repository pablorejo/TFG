import pandas as pd
from tqdm import tqdm

exit(0)
pd.set_option('display.max_colwidth', None)
# Reemplaza 'ruta_del_archivo.txt' con la ruta real de tu archivo
ruta_del_archivo_occurrences = 'ocurrencias.csv'

chunksize = 1 * 10 ** 6 
# Cargar el archivo como un DataFrame de Pandas

print("Leyendo el fichero de ocurrencias")
chunks = pd.read_csv(ruta_del_archivo_occurrences, delimiter=',',  chunksize=chunksize, low_memory=False, on_bad_lines='skip')

print("Leyendo el fichero multimedia")

# Inicializar una variable para seguir la pista de si estamos procesando el primer chunk
es_primer_chunk = True
nombre_archivo = 'ocurrencias_parseado.csv'

print("Iniciando chunks\n")
for df_occurrences in tqdm(chunks, desc="Procesando elementos", unit="elemento"):

    # Eliminar las filas con el campo "identifier" vacío
    df_occurrences = df_occurrences.dropna(subset=['identifier'])

    # print(df_reducido[df_reducido['acceptedScientificName'] == 'Anatina anatina (Spengler, 1802)'][['gbifID', 'acceptedScientificName', 'identifier']])

    # Guardar el chunk en el archivo CSV. Usa 'w' (escribir) para el primer chunk y 'a' (añadir) para los siguientes
    df_occurrences.to_csv(nombre_archivo, mode='w' if es_primer_chunk else 'a', index=False, header=es_primer_chunk)

    # Después del primer chunk, asegúrate de que header se establezca en False
    if es_primer_chunk:
        es_primer_chunk = False


print("El programa ha finalizado con éxito")


