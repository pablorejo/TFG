import os
import pandas as pd
from tqdm import tqdm
from conf import *
from defs import *
import threading
from queue import Queue
from defs_img import *
import ast

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
            os.makedirs(folder_path,exist_ok=True)
        
        url_list = ast.literal_eval(row['identifier'])

        index = 0
        for url_image in url_list:
            full_path = os.path.join(folder_path, f"{parse_name(str(row['gbifID']))}_{index}.jpg")
            webp_path = full_path.replace(".jpg", ".webp")
            if ((not os.path.exists(full_path) or is_corrupt_image(full_path)) and not os.path.exists(webp_path)):
                if download_image(url_image, full_path):
                    queue.put((full_path, row))
                    index += 1
    return 

def process_images(results):
    valid_rows = []
    model_discard = chek_model(DISCARD_MODEL_PATH)
    # model_crop = YOLO(chek_model(DETECT_MODEL_PATH))
    for full_path, row in results:
        try:
            if discard_bad_image(full_path,model_discard):
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
        result = queue.get()
        if result != (None,None):
            results.append(result)
    
    valid_rows = process_images(results)
    return valid_rows

def process_pandas():

    # You can consult the number of threads that you can execute with comand ulimit -u in linux, recomended use a chunk with less size than threads
    chunksize = 10**3
    empty_folder(TEMP_IMAGE_PATH)
    info(f"reading file {ALL_DATA_CSV}")
    df_occurrences = pd.read_csv(ALL_DATA_CSV, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
    
    # Calculate the number of chunks
    total_chunks = sum(1 for _ in pd.read_csv(ALL_DATA_CSV, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip'))
    info(f"Total number of chunks: {total_chunks}.\n")
    
    results = []
    for chunk in tqdm(df_occurrences,total=total_chunks,desc="Procesing chunks"):
        result = process_chunk(chunk)
        results.append(result)
    
    valid_rows = [row for sublist in results for row in sublist]
    df_valid_rows = pd.DataFrame(valid_rows)
    df_valid_rows.to_csv(PROCESSED_DATA_CSV, index=False)

if __name__ == "__main__":
    set_thread_priority()
    process_pandas()
