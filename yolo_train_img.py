# if __name__ == "__main__":
from conf import *
from tqdm import tqdm
import os
import pandas as pd
from random import randint
from defs import *
from requests.exceptions import RequestException
from queue import Queue
import threading
from defs_img import *
from playsound import playsound
import ast
import pickle
from multiprocessing import Process, Lock,cpu_count, Manager
from ultralytics import YOLO
import torch

STATE_FILE = 'yolo_train_img_data.pkl'

def save_state(state):
    with open(STATE_FILE, 'wb') as f:
        pickle.dump(state, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'rb') as f:
            return pickle.load(f)
    return None


def create_conf_yaml(path_file,path_to_train,clases):
    """
    This function create a yaml configuration

    Args:
        path_file (str): path do you want save this configuration
        path_to_train (str): path that you have save data training
        clases (list): string list that you have all classes that you whan clasify
    """

    with open(path_file,'w') as file:
        texto = f"""# Configurando el entrenamiento
path: {path_to_train} # dataset root dir
train: {training_data_path['train']}/ # train images (relative to 'path') 39384 images
val: {training_data_path['valid']}/ # val images (relative to 'path') 15062 images
test: {training_data_path['test']}/ # test images (optional) https://eval.ai/web/challenges/challenge-page/800/overview
nc: {len(clases)}
# Classes
names: {list(clases)} # class names
"""
        file.write(texto)
        file.close()

    return path_file
        
def process_row(
        row, 
        training, 
        initial_counts, 
        temp_image_path, 
        counts_with_crops, 
        counts_with_transformations_and_crops, 
        semaphore_values: threading.Semaphore,
        semaphore_models: threading.Semaphore,
        model_to_discard: YOLO,
        model_to_crop: YOLO):
    
    proceed = True

    if proceed:
        semaphore_values.acquire()
        if (counts_with_transformations_and_crops[row[training]]) < TOTAL_IMAGES_PER_CLASS  :  # If the taxon already has all the necessary images, do not download more.
            semaphore_values.release()
            
            # Check if the row has a valid identifier
            
            if pd.notna(row['identifier']):
                # Build the folder path based on the taxonomic classification
                folder_path = os.path.join(temp_image_path, parse_name(row[training]))

                # Create the folder if it does not exist
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path, exist_ok=True)
                    info('Folder created successfully')

                url_list = ast.literal_eval(row['identifier'])
        
                index = 0
                for url_image in url_list:
                    # Build the full path of the image file to save
                    full_path = os.path.join(folder_path, f"{parse_name(str(row['gbifID']))}_{index}.jpg")

                    # Download and save the image if it does not exist yet
                    if  (not is_corrupt_image(full_path)):
                        try:
                            if download_image(url_image, full_path):
                                process_image(
                                    full_path,
                                    row,
                                    counts_with_crops,
                                    counts_with_transformations_and_crops,
                                    training,
                                    initial_counts,
                                    model_to_discard,
                                    model_to_crop,
                                    semaphore_values,
                                    semaphore_models
                                )

                        except KeyboardInterrupt:
                            fail("Program terminated with errors successfully")
                            exit(-1)

                        except RequestException as e:
                            warning(f"URL error: " + row['identifier'] + f"\nError: {e}")

                        except Exception as e:
                            warning(f"Error: {e}")
                            pass
        else:
            semaphore_values.release()
    return

def process_image(
        full_path, row , 
        counts_with_crops, 
        counts_with_transformations_and_crops, 
        training, 
        initial_counts,
        model_to_discard: YOLO,
        model_to_crop: YOLO,
        semaphore_values: threading.Semaphore, 
        semaphore_models: threading.Semaphore
        ):
    
    
    with semaphore_models:
        discard = discard_bad_image(full_path,model_to_discard)

    if discard:
        number_of_transformations = TRANSFORMATIONS_PER_IMAGE
        total_transformations = None
        with semaphore_values:

            not_processing = TOTAL_IMAGES_PER_CLASS - counts_with_transformations_and_crops[row[training]]
            if not_processing <= 0:
                os.remove(full_path)
                return
            
            initial_counts[row[training]] += 1
            with semaphore_models:
                paths = crop_images(src_img=full_path,model_to_crop=model_to_crop,model_to_discard=model_to_discard)
                number_of_crops = len(paths)
                total = number_of_crops + number_of_crops * TRANSFORMATIONS_PER_IMAGE
                
                if not_processing - total < 0:
                    if not_processing - number_of_crops < 0:
                        for i in range(number_of_crops - not_processing):
                            os.remove(paths[i])
                        number_of_transformations = 0
                        
                        counts_with_crops[row[training]] += len(paths)
                        counts_with_transformations_and_crops[row[training]] += len(paths)
                    else:
                        total_transformations = not_processing - number_of_crops
                        
                        counts_with_crops[row[training]] += len(paths)
                        counts_with_transformations_and_crops[row[training]] += len(paths) + total_transformations
                else:
                    counts_with_crops[row[training]] += len(paths)
                    counts_with_transformations_and_crops[row[training]] += len(paths) + len(paths) * TRANSFORMATIONS_PER_IMAGE
            
        
        new_paths = []
        for path in paths:
            new_paths.append(convert_to_webp(path))
        
        break_bool = False
        transformations = 0
        for k in range(number_of_transformations):
            for new_path in new_paths:
                augment_image(new_path, k)
                transformations += 1
                if total_transformations != None and transformations >= total_transformations:
                    break_bool = True
                    break
            if break_bool:
                break
    else:
        pass
        # os.remove(full_path)

            
def process_chunk(
        chunk, 
        training, 
        initial_counts, 
        counts_with_crops, 
        counts_with_transformations_and_crops,
        temp_image_path: str,
        model_to_discard: YOLO,
        model_to_crop: YOLO):
    
    threads = []

    
    semaphore_values = threading.Semaphore(1)
    semaphore_models = threading.Semaphore(1)

    
    for _, row in pd.DataFrame(chunk).iterrows():
        if USE_THREADS_TO_DOWNLOAD:
            
            thread = threading.Thread(target=process_row, 
                args=(
                    row, 
                    training, 
                    initial_counts, 
                    temp_image_path, 
                    counts_with_crops, 
                    counts_with_transformations_and_crops, 
                    semaphore_values,
                    semaphore_models,
                    model_to_discard,
                    model_to_crop,
                )
            )
            threads.append(thread)
            thread.start()
        else:
        
            process_row(row, 
                    training, 
                    initial_counts, 
                    temp_image_path, 
                    counts_with_crops, 
                    counts_with_transformations_and_crops, 
                    semaphore_values,
                    semaphore_models,
                    model_to_discard,
                    model_to_crop)

        
    if USE_THREADS_TO_DOWNLOAD:
        for thread in threads:
            thread.join()
  
    return 

def calculate_data_of_df(data_path: str,filter_colums=None, filters=None):
    info('Calculating model counts')
    model_count_dict = {}
    
    chunksize = 10**3

    first_time = True
    total_lines = 0

    for column_name in taxonomic_ranks:
        df = pd.read_csv(data_path, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
        unique_values = set()
        for chunk in df:
            for colum, filter_values in zip(filter_colums,filters):
                index_of_filter = taxonomic_ranks.index(colum)
                if index_of_filter <= taxonomic_ranks.index(column_name):
                    chunk = chunk[chunk[colum].isin(filter_values)]


            unique_values.update(chunk[column_name].unique()) 

            if first_time:
                total_lines += len(chunk)

        first_time = False
        model_count_dict[column_name] = len(unique_values)
        info(f"\nIn the taxon of {column_name} we have a total of: {model_count_dict[column_name]}")
        info(f"There names of the diferents {column_name} are:\n{unique_values}\n")

    model_count_dict['total_lines'] = total_lines
    info(f"There are {total_lines} lines in this data frame")
    return model_count_dict
    

taxonomic_ranks = [
    'class',
    'order',
    'family',
    'genus',
    'acceptedScientificName'
]

def initial_df_processing(df_occurrences, training, column_filters, filters,taxon_index):

    manager = Manager()

    total_counts = manager.dict() # Contains a dictionary of the taxon name with the number of existing data points.
    
    initial_counts = manager.dict()  # Contains a dictionary of the taxon name and keeps track of how many there are of each.
    dfs = []
    
    for chunk in df_occurrences:
        if training and len(column_filters) == len(filters):
            # Apply each specified column filter
            
            for col_name, filter_value in zip(column_filters, filters):
                index_of_filter = taxonomic_ranks.index(col_name)
                if index_of_filter <= taxon_index:
                    chunk = chunk[chunk[col_name].isin(filter_value)]
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
            total_counts[keys[i]]+= values[i]
        
        dfs.append(chunk)  # Add the chunk
    return initial_counts, total_counts, dfs

def no_more_images(counts: dict):
    for _, value in counts.items():
        if value < TOTAL_IMAGES_PER_CLASS:
            return False
    return True

def increase_images(initial_counts,counts_with_crops):
    # Increase images if necessary
    for key, value in initial_counts.items():
        valor = value + counts_with_crops[key]
        if valor < IMAGE_SAMPLE_COUNT:
            folder_path = f"{TEMP_IMAGE_PATH}/{key}/"
            images = find_images(folder_path)
            if len(images) > 0:
                new_counter = 0
                while len(images) + new_counter < IMAGE_SAMPLE_COUNT:
                    augment_image(images[randint(0, len(images)-1)], value)
                    new_counter += 1
            else:
                warning(f"No images found for class: {key}")

def filter_chunk(chunk: pd.DataFrame,  counts_with_transformations_and_crops: dict, training: str):
    
    values_completed = [key for key, value in counts_with_transformations_and_crops.items() if value >= TOTAL_IMAGES_PER_CLASS]
    
    # Filtrar el chunk eliminando las filas donde 'training' esté en 'values_completed'
    filtered_chunk = chunk[~chunk[training].isin(values_completed)]
    return filtered_chunk,no_more_images(counts_with_transformations_and_crops)

def train_model(model, train_folder_path, model_name, start_time_func, execution_time_process_chunk, model_folder):
    start_time_train = time.time()
    
    train_yolo_model(model=model,model_name=model_name, train_folder_path=train_folder_path,model_folder=model_folder)
    
    end_time_func = time.time()
    execution_time_func = end_time_func - start_time_func
    execution_time_train = end_time_func - start_time_train

    save_information(model_folder, execution_time_func, execution_time_train, execution_time_process_chunk)
    return "Train Completed"
    
def save_information(model_folder,execution_time_func,execution_time_train,execution_time_process_chunk):
    end_time_func = time.time()
    # play notification sound
    playsound('extras/noti.mp3')
    info(f"The function took {execution_time_func} seconds to execute.")
    info(f"The train took {execution_time_train} seconds to execute.")
    info(f"The proces of chunks took {execution_time_process_chunk} seconds to execute.")
    
    with open(os.path.join(model_folder,'information.txt'), 'w') as file:
        file.write(f"The function took {execution_time_func} seconds to execute.\n")
        file.write(f"The train took {execution_time_train} seconds to execute.\n")
        file.write(f"The proces of chunks took {execution_time_process_chunk} seconds to execute.\n")
        file.close()

def train(
    model_folder: str = os.path.join(PATH_MODELS_TRAINED,"model_g"),
    path_model_to_train = MODEL_INIT,
    column_filters: list = [],
    filters: list = [],
    taxon_index: int = 0,
    temp_image_path = TEMP_IMAGE_PATH,
    train_folder_path = TRAINING_DEST_PATH,
    resume = False,
    download_images_bool = True,
    ):

    start_time_func = time.time()

    if resume:
        state = load_state()
        if state:
            initial_model_name = state['initial_model_name']
            column_filters = state['column_filters']
            filters = state['filters']
            taxon_index = state['taxon_index']
            temp_image_path = state['temp_image_path']
            train_folder_path = state['train_folder_path']
            initial_counts = state['initial_counts']
            total_counts = state['total_counts']
            counts_with_crops = state['counts_with_crops']
            counts_with_transformations_and_crops = state['counts_with_transformations_and_crops']
        else:
            raise ValueError("No saved state to resume from.")
    
    if download_images_bool:
        empty_folder(temp_image_path)  # Empty the temporary training image folder.
    training = taxonomic_ranks[taxon_index]
    
    chunksize = 10**3
    df_occurrences = pd.read_csv(PROCESSED_DATA_CSV, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
    

    start_time_proces_chunk = time.time()

    if download_images_bool:
        # Load previous state if resuming
        initial_counts, total_counts, dfs = initial_df_processing(df_occurrences, training, column_filters, filters,taxon_index)
        
        # Get previous chunk data and filter it

        manager = Manager()
        counts_with_crops = manager.dict(initial_counts.copy())
        counts_with_transformations_and_crops = manager.dict(initial_counts.copy()) 

        # Shuffle dataframe if required
        if SHUFFLE_DATAFRAME and taxon_index == 0:
            info('Shuffling dataframe')
            dfs = shuffle_DataFrame(dfs)

        
        info(f'Start processing chunks for model: {os.path.split(model_folder)[-1]}')

        start_time_proces_chunk = time.time()


        with tqdm(total=len(dfs)) as pbar:
            model_to_discart = YOLO(DISCARD_MODEL_PATH)
            model_to_crop = YOLO(DETECT_MODEL_PATH)

            i = 0
            for chunk in dfs:
                chunk_2,end_of_images = filter_chunk(chunk,counts_with_transformations_and_crops,training)
                if end_of_images:
                    break
                process_chunk(chunk_2, 
                    training, 
                    initial_counts, 
                    counts_with_crops, 
                    counts_with_transformations_and_crops,
                    temp_image_path, 
                    model_to_discart, 
                    model_to_crop)
                i+=1
                
        # Increase images if needed
        info('cheking data images and increasing')
        increase_images(initial_counts,counts_with_transformations_and_crops)
        
    end_time_proces_chunk = time.time()
    

    # model_name = f"{initial_model_name}_{column_filters[-1]}_{filters[-1]}" if column_filters else initial_model_name
    model_name = os.path.split(model_folder)[-1]

    
    info(MODEL_INIT)
    chek_folder(model_folder)
    if download_images_bool:
        if os.path.exists(model_folder):
            empty_folder(model_folder)
            os.rmdir(model_folder)
            

        info(f"""Using the following image counts
            Downloaded images: {initial_counts}
            Cropped images: {counts_with_crops}
            Total images: {counts_with_transformations_and_crops}""")
    
    # Copy temporary images to training folder, if empty, do not train
    if download_images_bool:
        traing_bool = copy_to_training(temp_image_path,train_folder_path)
    else:
        traing_bool = True
        
    if traing_bool:
        info("Training: " + model_name)
        execution_time_process_chunk = end_time_proces_chunk - start_time_proces_chunk

        model = chek_model(path_model_to_train)
        if model == None:
            model = chek_model(MODEL_INIT)
        # if __name__ == "__main__":
        train_model(model, 
            train_folder_path, 
            model_name,
            start_time_func,
            execution_time_process_chunk,
            model_folder)
        
        if download_images_bool:
            state = {
                    'model_folder': model_folder,
                    'column_filters': column_filters,
                    'filters': filters,
                    'taxon_index': taxon_index,
                    'temp_image_path': temp_image_path,
                    'train_folder_path': train_folder_path,
                    'initial_counts': initial_counts,
                    'total_counts': total_counts,
                    'counts_with_crops': counts_with_crops,
                    'counts_with_transformations_and_crops': counts_with_transformations_and_crops
                }
            save_state(state)

        # If we reach the species identification step, the taxonomic rank will be the last in the list. Save the filter list to a txt file for result analysis.
        if taxonomic_ranks[taxon_index] == taxonomic_ranks[-1]:
            with open(os.path.join(model_folder,'data.txt'), 'w') as file:
                for filter_item in filters:
                    for filter in filter_item:
                        file.write(filter + ",")     
                file.close()
            

        path_to_model = os.path.join(model_folder,'weights','best.pt') 

        second_loop = 0

        if download_images_bool:
            empty_folder(temp_image_path)

        
        for key, _ in total_counts.items():
            if taxon_index == 0 and second_loop == 1:
                info("second loop")
            
            if taxon_index < len(taxonomic_ranks) - 1:
                next_column_filters = column_filters.copy()
                next_filters = filters.copy()

                traing_bool = True
                if taxonomic_ranks[taxon_index] not in next_column_filters:
                    next_column_filters.append(taxonomic_ranks[taxon_index])
                    next_filters.append([key])
                else:
                    index_filter = next_column_filters.index(taxonomic_ranks[taxon_index])
                    if key in next_filters[index_filter]:
                        next_filters[index_filter] = [key]
                    else:
                        traing_bool = False
                

                if traing_bool:
                    next_model_folder = os.path.join(model_folder,f"{taxonomic_ranks[taxon_index]}_{key}")
                    train(
                        model_folder=next_model_folder,
                        column_filters=next_column_filters,
                        filters=next_filters,
                        taxon_index=taxon_index+1,
                        path_model_to_train=path_to_model)

            else:
                info(f"Finished {column_filters} of {filters}")
            second_loop = 1
    else:
        text = f"No data exists for these filters\n {column_filters}\n{filters}"
        with open(os.path.join(model_folder,'info.txt'),'w') as file:
            file.write(text)
        info(text)

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
        filter_colums=[
            taxonomic_ranks[0],
            taxonomic_ranks[1],
            taxonomic_ranks[2],
            taxonomic_ranks[3],
            taxonomic_ranks[4]
            ]
        
        filters=[
            ['Gastropoda', 'Bivalvia', 'Cephalopoda', 'Monoplacophora', 'Polyplacophora', 'Scaphopoda'],
            ['Cyrtodontida', 'Arcida', 'Sphaeriida', 'Venerida'],
            ['Arcidae', 'Pichleriidae', 'Cardiolidae', 'Glycymerididae', 'Anatinellidae', 'Veneridae'],
            ['Artena', 'Gratelupia', 'Pelecyora', 'Bassina', 'Atamarcia'],
            ['Antigona inca Olsson, 1939', 'Bassina disjecta (Perry, 1811)', 'Pelecyora corculum (Römer, 1870)', 'Antigona neglecta Clark, 1918', 'Pelecyora hatchetigbeensis (Aldrich, 1886)', 'Bassina yatei (Gray, 1835)']
            ]
        
        taxon_index = 0
        # calculate_data_of_df(PROCESSED_DATA_CSV,filter_colums=filter_colums,filters=filters)
        train(column_filters=filter_colums,filters=filters,taxon_index=taxon_index,train_folder_path=TRAINING_DEST_PATH,download_images_bool=True)
