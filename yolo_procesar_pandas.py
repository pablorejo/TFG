import os
import pandas as pd
from tqdm import tqdm
from conf import *
from defs import *
import threading
from queue import Queue
from defs_img import *
import ast
from multiprocessing import Pool

taxonomic_ranks = [
    'class',
    'order',
    'family',
    'genus',
    'acceptedScientificName'
]

def process_row(row, semaphore_models, semaphore_max_threads, model_discard, result_queue: Queue):
    
    with semaphore_max_threads:
        if pd.notna(row['identifier']):
            folder_path = os.path.join(TEMP_IMAGE_PATH, 'test')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path,exist_ok=True)
            
            url_list = ast.literal_eval(row['identifier'])

            index = 0
            good_images =  []
            for url_image in url_list:
                full_path = os.path.join(folder_path, f"{parse_name(str(row['gbifID']))}_{index}.jpg")
                if ((not os.path.exists(full_path) or is_corrupt_image(full_path))):
                    if download_image(url_image, full_path):
                        good_image =  process_image(
                                full_path,model_discard, semaphore_models
                            )
                        if os.path.exists(full_path):
                            os.remove(full_path)
                            
                        if good_image:
                            good_images.append(url_image)
                            
                        index += 1
            if len(good_images) > 0:
                row['identifier'] = good_images
                if USE_THREADS_TO_DOWNLOAD:
                    result_queue.put(row)
                else:
                    return row
    return 

def process_image(full_path,model_discard, semaphore_models):
    # model_crop = YOLO(chek_model(DETECT_MODEL_PATH))  
    try:
        semaphore_models.acquire()
        if discard_bad_image(full_path,model_discard):
            semaphore_models.release()
            return True
        semaphore_models.release()
        if os.path.exists(full_path):
            os.remove(full_path)
    except Exception as e:
        warning(f"Error in process_images: {e}")
    return False

def process_chunk(chunk):
    info('Processing chunk')
    results = []
    threads = []
    
    semaphore_max_threads = threading.Semaphore(MAX_THREADS_DOWNLOADING_PER_PROCESS)
    semaphore_models = threading.Semaphore(1)
    model_discard = chek_model(DISCARD_MODEL_PATH)
    
    result_queue = Queue()
    
    if USE_PROCESS_TO_DOWNLOAD:
        for _, row in chunk.iterrows():
            thread = threading.Thread(target=process_row, args=(row, semaphore_models, semaphore_max_threads, model_discard, result_queue))
            threads.append(thread)
            thread.start()
            
        
    
        valid_rows = []
        for thread in threads:
            thread.join()
        
        
        valid_rows = []
        while not result_queue.empty():
            valid_rows.append(result_queue.get())
    else:
        valid_rows = []
        for _, row in chunk.iterrows():
            valid_rows.append(process_row(row, semaphore_models, semaphore_max_threads, model_discard, result_queue))
        
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
    
    time_init_process = time.time()
    if USE_PROCESS_TO_DOWNLOAD:
        with Pool(processes=NUMBER_OF_PROCESS_PANDAS) as pool_process_pandas:
            results = pool_process_pandas.map(process_chunk,df_occurrences)
    else:
        for chunk in tqdm(df_occurrences,total=total_chunks,desc="Procesing chunks"):
            result = process_chunk(chunk)
            results.append(result)
    
    valid_rows = [row for sublist in results for row in sublist]
    df_valid_rows = pd.DataFrame(valid_rows)
    time_end_process = time.time()
    time_to_process = time_end_process - time_init_process
    text_info = f'Yolo process pandas took {time_to_process}'
    info(text_info)
    with open(os.path.join(PANDAS_CSV_PATH,'info_yolo_process_pandas.txt'),'w') as file:
        file.write(text_info)
    
    df_valid_rows.to_csv(PROCESSED_DATA_CSV, index=False)

if __name__ == "__main__":
    set_thread_priority()
    process_pandas()
