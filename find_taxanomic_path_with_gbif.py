import pandas as pd
import conf

def find_taxonomic_path(gbif_id: str, data_frame):
    return data_frame[data_frame['gbifID'] == int(gbif_id)]

if __name__ == "__main__":
    gbif_id = input("put gbif id: ").strip()
    print(f"buscando en __{gbif_id}__ ...")
    chunksize = 10**3
    
    # Leer el archivo en trozos
    df_occurrences = pd.read_csv(conf.PROCESSED_DATA_CSV, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
    
    # Iterar sobre cada trozo de datos
    for chunk in df_occurrences:
        row = find_taxonomic_path(gbif_id, chunk)
        
        # Verificar si se encontró alguna fila
        if not row.empty:
            break
    if not row.empty:
        for index, row_data in row.iterrows():
            for column, value in row_data.items():
                print(f"{column}: {value}")
    else:
        print("No se encontró ninguna fila con el gbifID proporcionado.")
