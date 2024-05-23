import os
import pandas as pd
from tqdm import tqdm
from conf import *
from defs import *
import threading
from queue import Queue
from defs_img import *

taxonomic_ranks = [
    'class',
    'order',
    'family',
    'genus',
    'acceptedScientificName'
]

def process_row(row, queue: Queue):
    if pd.notna(row['identifier']):
        folder_path = os.path.join(TEMP_IMAGE_PATH, 'test')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        full_path = os.path.join(folder_path, parse_name(str(row['gbifID'])) + ".jpg")
        webp_path = full_path.replace(".jpg", ".webp")

        if ((not os.path.exists(full_path) or is_corrupt_image(full_path)) and not os.path.exists(webp_path)):
            if download_image(row['identifier'], full_path):
                queue.put((full_path, row))
                return 
    
    queue.put((None, None))     
    return 

def process_images(results):
    valid_rows = []
    for full_path, row in results:
        try:
            if discard_bad_image(full_path):
                valid_rows.append(row)
            if os.path.exists(full_path):
                os.remove(full_path)
        except Exception as e:
            warning(f"Error in process_images: {e}")
    return valid_rows

def process_chunk(chunk):
    info('Processing chunk')
    results = []
    queue = Queue()
    threads = []
    for _, row in chunk.iterrows():
        thread = threading.Thread(target=process_row, args=(row, queue))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
        
    while not queue.empty():
        results.append(queue.get())
    
    valid_rows = process_images(results)
    return valid_rows

def process_pandas():
    chunksize = 10**3
    empty_folder(TEMP_IMAGE_PATH)
    df_occurrences = pd.read_csv(ALL_DATA_CSV, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
    
    # Calculate the number of chunks
    total_chunks = sum(1 for _ in pd.read_csv(ALL_DATA_CSV, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip'))
    info(f"Total number of chunks: {total_chunks}.\n")
    
    results = []
    with tqdm(total=total_chunks) as pbar:
        for chunk in df_occurrences:
            result = process_chunk(chunk)
            results.append(result)
            pbar.update()
    
    valid_rows = [row for sublist in results for row in sublist]
    df_valid_rows = pd.DataFrame(valid_rows)
    df_valid_rows.to_csv(PROCESSED_DATA_CSV, index=False)

if __name__ == "__main__":
    set_thread_priority()
    process_pandas()
