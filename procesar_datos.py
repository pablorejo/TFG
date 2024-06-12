from conf import *
from os import path
import pandas as pd
from tqdm import tqdm

def process_multimedia():
    info('Processing multimedia...')
    df_images = pd.read_csv(IMAGE_DATA, delimiter="\t", chunksize=10**4, low_memory=False, on_bad_lines='skip')
    chunks = []
    multimedia_columns = ['gbifID', 'identifier']
    for chunk in tqdm(df_images):
        chunk_2 = chunk[multimedia_columns]
        chunks.append(chunk_2.dropna())
        
    info('concat chunks')
    df_images_concatenated = pd.concat(chunks, ignore_index=True)
    
    info('group by gbifID and applying list to identifier column')
    # Agrupar por 'gbifID' y crear listas de 'identifier'
    df_images_grouped = df_images_concatenated.groupby('gbifID')['identifier'].apply(list).reset_index()
    
    info(df_images_grouped.head())
    return df_images_grouped

def process_occurrences():
    df_occurrences = pd.read_csv(OCCURRENCE_DATA, delimiter="\t", chunksize=10**4, low_memory=False, on_bad_lines='skip')
    chunks = []
    
    occurrence_columns = [
        'gbifID',
        'class',
        'order',
        'family',
        'genus',
        'acceptedScientificName',
        'iucnRedListCategory'
    ]
    info('Processing occurrences...')
    for chunk in tqdm(df_occurrences):
        chunk_2 = chunk[occurrence_columns]
        chunks.append(chunk_2.dropna())
    
    info('concat chunks')
    df_occurrences_concatenated = pd.concat(chunks, ignore_index=True)
    info(df_occurrences_concatenated.head())
    return df_occurrences_concatenated

def main():
    df_images = process_multimedia()
    df_occurrences = process_occurrences()
    
    # Merge the DataFrames by the 'gbifID' column
    info('Merging data')
    merged_df = pd.merge(df_occurrences, df_images, on='gbifID', how='inner')
    
    # Remove rows that contain any empty fields
    info('Removing rows with empty columns')
    cleaned_df = merged_df.dropna()
    
    # Remove rows where 'iucnRedListCategory' is 'EX', i.e., those of extinct species
    info('Removing extinct species')
    final_df = cleaned_df[cleaned_df['iucnRedListCategory'] != 'EX']
    
    info(final_df.head())
    info('Saving data')
    final_df.to_csv(ALL_DATA_CSV, index=False)
    reduced_csv = ALL_DATA_CSV.split('.')[0] + "_reduced.csv"
    final_df.head().to_csv(reduced_csv, index=False)

if __name__ == '__main__':
    set_thread_priority()
    main()
