from conf import *
from ultralytics import YOLO
from tqdm import tqdm
import os
import pandas as pd
from random import randint
from defs import *
from requests.exceptions import RequestException
from queue import Queue
import threading
from defs_img import *

def process_row(row, training, initial_counts, queue):
    proceed = True

    if proceed:
        if row[training] not in initial_counts:
            initial_counts[row[training]] = 0
                    
        if initial_counts[row[training]] < IMAGE_SAMPLE_COUNT:  # If the taxon already has all the necessary images, do not download more.
            # Check if the row has a valid identifier
            if pd.notna(row['identifier']):
                # Build the folder path based on the taxonomic classification
                folder_path = os.path.join(TEMP_IMAGE_PATH, parse_name(row[training]))

                # Create the folder if it does not exist
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path, exist_ok=True)
                    info('Folder created successfully')

                # Build the full path of the image file to save
                full_path = os.path.join(folder_path, parse_name(str(row['gbifID'])) + ".jpg")
                webp_path = full_path.replace(".jpg", ".webp")

                # Download and save the image if it does not exist yet
                if ((not os.path.exists(full_path) or is_corrupt_image(full_path)) and not os.path.exists(webp_path)):
                    try:
                        if download_image(row['identifier'], full_path):
                            initial_counts[row[training]] += 1  # Add one to this row
                            queue.put((full_path, row))
                            return

                    except KeyboardInterrupt:
                        fail("Program terminated with errors successfully")
                        exit(-1)

                    except RequestException as e:
                        warning(f"URL error: " + row['identifier'] + f"\nError: {e}")

                    except Exception as e:
                        warning(f"Error: {e}")
                        pass
    return

def process_images(results, counts_with_crops, counts_with_transformations_and_crops, training, initial_counts):
    # initial_counts = dict.fromkeys(initial_counts,0)  # Set all dictionary keys to 0
    
    for full_path, row in results:
        if discard_bad_image(full_path):
            paths = crop_images(src_img=full_path)
            for path in paths:
                new_path = convert_to_webp(path)
                for k in range(TRANSFORMATIONS_PER_IMAGE):
                    augment_image(new_path, k)
            
            counts_with_crops[row[training]] += len(paths)
            counts_with_transformations_and_crops[row[training]] += len(paths) + len(paths) * TRANSFORMATIONS_PER_IMAGE
        else:
            initial_counts[row[training]] -= 1
            
def process_chunk(chunk, training, initial_counts, counts_with_crops, counts_with_transformations_and_crops):
    
    queue = Queue()
    threads = []
    results = []
    for _, row in pd.DataFrame(chunk).iterrows():
        thread = threading.Thread(target=process_row, args=(row, training, initial_counts, queue))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
        
    while not queue.empty():
        results.append(queue.get())
    
    process_images(results, counts_with_crops, counts_with_transformations_and_crops, training, initial_counts)
    return 

def calculate_model_counts(data_path: str):
    info('Calculating model counts')
    model_count_dict = {}
    
    chunksize = 10**3

    for column_name in taxonomic_ranks:
        df = pd.read_csv(PROCESSED_DATA_CSV, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
        unique_values = set()
        for chunk in df:
            unique_values.update(chunk[column_name].unique()) 
        
        model_count_dict[column_name] = len(unique_values)
        info(f"In the range of {column_name} we have a total of: {model_count_dict[column_name]}\n")
    

taxonomic_ranks = [
    'class',
    'order',
    'family',
    'genus',
    'acceptedScientificName'
]

def initial_df_processing(df_occurrences, training, column_filters, filters):
    total_counts = {}  # Contains a dictionary of the taxon name with the number of existing data points.
    
    initial_counts = {}  # Contains a dictionary of the taxon name and keeps track of how many there are of each.
    dfs = []
    
    for chunk in df_occurrences:
        if training and len(column_filters) == len(filters):
            # Apply each specified column filter
            for col_name, filter_value in zip(column_filters, filters):
                chunk = chunk[chunk[col_name] == filter_value]
        values = []
        keys = []
        
        for value in chunk[training].value_counts().values:
            values.append(value)
        for key in chunk[training].value_counts().keys().values:
            keys.append(key)
        
        for i in range(len(values)):
            if keys[i] not in total_counts:
                total_counts[keys[i]] = 0
                initial_counts[keys[i]] = 0
            total_counts[keys[i]] += values[i]
        
        dfs.append(chunk)  # Add the chunk
    return initial_counts, total_counts, dfs

def no_more_images(counts: dict):

    for key, value in counts.items():
        if value < IMAGE_SAMPLE_COUNT:
            return False
    return True

def increase_images(initial_counts):
    # Increase images if necessary
    for key, value in initial_counts.items():
        if value < IMAGE_SAMPLE_COUNT:
            folder_path = f"{TEMP_IMAGE_PATH}/{key}/"
            images = find_images(folder_path)
            if len(images) > 0:
                new_counter = 0
                while len(images) + new_counter < IMAGE_SAMPLE_COUNT:
                    augment_image(images[randint(0, len(images)-1)], value)
                    new_counter += 1
            else:
                warning(f"No images found for class: {key}")

def filter_chunk(chunk: pd.DataFrame, initial_counts: dict, training: str):
    values_completed = [key for key, value in initial_counts.items() if value >= IMAGE_SAMPLE_COUNT]
    
    # Filtrar el chunk eliminando las filas donde 'training' esté en 'values_completed'
    filtered_chunk = chunk[~chunk[training].isin(values_completed)]
    return filtered_chunk

def train(
    initial_model_name: str = "model",
    model = YOLO('yolov8n-cls.pt'),
    column_filters: list = [],
    filters: list = [],
    taxon_index: int = 0):

    empty_folder(TEMP_IMAGE_PATH)  # Empty the temporary training image folder.
    training = taxonomic_ranks[taxon_index]
    
    chunksize = 10**2
    df_occurrences = pd.read_csv(PROCESSED_DATA_CSV, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
    
    # if VERBOSE :
    #     calculate_model_counts(df_occurrences)
    # exit(0)
    
    # Get previous chunk data and filter it
    initial_counts, total_counts, dfs = initial_df_processing(df_occurrences, training, column_filters, filters)

    counts_with_crops = initial_counts.copy()
    counts_with_transformations_and_crops = initial_counts.copy()
    # If the dataframe should be shuffled
    if SHUFFLE_DATAFRAME:
        info('Shuffling dataframe')
        dfs = shuffle_DataFrame(dfs)
    
    info('Start processing chunks')
    with tqdm(total=len(dfs)) as pbar:
        for chunk in dfs:
            filtered_chunk = filter_chunk(chunk,initial_counts,training)
            process_chunk(filtered_chunk, training, initial_counts, counts_with_crops, counts_with_transformations_and_crops)
            if no_more_images(initial_counts):
                break
            pbar.update()
                
    # Increase images if needed
    info('cheking data images and increasing')
    increase_images(initial_counts)
    

    model_name = f"{initial_model_name}_{column_filters[-1]}_{filters[-1]}" if column_filters else initial_model_name
    model_folder = f'runs/classify/{model_name}'

    if os.path.exists(model_folder):
        empty_folder(model_folder)
        os.rmdir(model_folder)


    info(f"""Using the following image counts
        Downloaded images: {initial_counts}
        Cropped images: {counts_with_crops}
        Total images: {counts_with_transformations_and_crops}""")
    
    # Copy temporary images to training folder, if empty, do not train
    if copy_to_training(TEMP_IMAGE_PATH): 
        info("Training: " + model_name)
        results = model.train(epochs=TRAIN_EPOCHS, imgsz=IMAGE_SIZE, name=model_name)
        
        # If we reach the species identification step, the taxonomic rank will be the last in the list. Save the filter list to a txt file for result analysis.
        if taxonomic_ranks[taxon_index] == taxonomic_ranks[-1]:
            with open(os.path.join(model_folder,'data.txt'), 'w') as file:
                for filter_item in filters:
                    file.write(filter_item + ",")     
                
        model = YOLO(os.path.join(model_folder, 'weights', 'best.pt'), task='')

        second_loop = 0
        for key, value in total_counts.items():
            if taxon_index == 0 and second_loop == 1:
                print("second loop")
            
            if taxon_index < len(taxonomic_ranks) - 1:
                next_column_filters = column_filters.copy()
                next_column_filters.append(taxonomic_ranks[taxon_index])
                next_filters = filters.copy()
                next_filters.append(key)
                train(column_filters=next_column_filters, filters=next_filters, taxon_index=taxon_index+1, model=model)
            else:
                info(f"Finished {column_filters} of {filters}")
            second_loop = 1
    else:
        info(f"No data exists for these filters\n {column_filters}\n{filters}")

def train_folder(model_name: str, folder_name: str):
    """Train data from a folder and its subfolders.
    
    Args:
    model_name: the name with which the model will be saved
    folder_name: the path to the folder to train"""
    # Train everything at once
    directories = get_directories(folder_name)

    for class_name in directories:
        images = find_images(f'{folder_name}/{class_name}')
        copy_to_training(class_type=class_name, lines=images)

    model = YOLO('yolov8n-cls.pt') 
    model.val(imgsz=IMAGE_SIZE)
    results = model.train(data='training.yaml', epochs=TRAIN_EPOCHS, imgsz=IMAGE_SIZE, name=model_name)


if __name__ == "__main__":
    set_thread_priority()
    info('Starting training')
    if TRAIN_LOCAL:
        for path in training_data_path.values():
            empty_folder(path)

        directories = get_directories(IMAGES_FOLDER)
        train_folder(taxonomic_ranks[0][0], IMAGES_FOLDER)

        for taxon, index in taxonomic_ranks:
            for name in get_folders_by_level(IMAGES_FOLDER, max_level=index):
                directories = get_directories(IMAGES_FOLDER)
                name = str(name).split("/")[-1]
                model_name = f"{taxon}_{name}"
                train_folder(model_name, name)
    else:
        train()
