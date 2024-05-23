"""
This file processes the multimedia and occurrence data to merge them all into a single file
so that the file will only contain the following columns:
gbif, class, order, family, genus, species, identifier (the URL to the image), iucnRedListCategory (to know if the species is extinct or not)
"""
from conf import *
from os import path
import pandas as pd

IMAGE_DATA = path.join(DISCARD_TXT_PATH, 'multimedia.txt')
OCCURRENCE_DATA = path.join(DISCARD_TXT_PATH, 'occurrence.txt')

def process_multimedia():
    info('Processing multimedia...')
    df_images = pd.read_csv(IMAGE_DATA, delimiter="\t", chunksize=10**4, low_memory=False, on_bad_lines='skip')
    chunks = []
    multimedia_columns = ['gbifID', 'identifier']
    for chunk in df_images:
        chunk_2 = chunk[multimedia_columns]
        chunks.append(chunk_2)
        
    df_images_concatenated = pd.concat(chunks, ignore_index=True)
    info(df_images_concatenated.head())
    return df_images_concatenated

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
    for chunk in df_occurrences:
        chunk_2 = chunk[occurrence_columns]
        chunks.append(chunk_2)
    
    df_occurrences_concatenated = pd.concat(chunks, ignore_index=True)
    info(df_occurrences_concatenated.head())
    return df_occurrences_concatenated
        
if __name__ == '__main__':
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
    
    info(cleaned_df.head())
    info('Saving data')
    cleaned_df.to_csv(ALL_DATA_CSV)
    reduced_csv = ALL_DATA_CSV.split('.')[0] + "_reduced.csv"
    cleaned_df.head().to_csv(reduced_csv)
