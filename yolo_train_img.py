from conf import *
import os
import pandas as pd
from defs import *
from requests.exceptions import RequestException
import threading
from defs_img import *
import ast
from multiprocessing import Manager, Pool
from ultralytics import YOLO
from save_context import *
from multiprocessing import Value

def process_row(
        taxon_index, 
        row, 
        training, 
        initial_counts, 
        temp_image_path, 
        counts_with_crops,
        counts_with_transformations_and_crops, 
        semaphore_values,
        semaphore_models: threading.Semaphore,
        semaphore_max_threads: threading.Semaphore,
        model_to_discard: YOLO,
        model_to_crop: YOLO):

    try:
        semaphore_max_threads.acquire()

        semaphore_values.acquire()
        try:
            if counts_with_transformations_and_crops[row[training]] >= total_image_per_cat(taxon_index):
                return
        finally:
            semaphore_values.release()

        if pd.notna(row['identifier']):
            # Build the folder path based on the taxonomic classification
            folder_path = os.path.join(temp_image_path, parse_name(row[training]))

            # Create the folder if it does not exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
                info('Folder created successfully')

            url_list = ast.literal_eval(row['identifier'])

            for index, url_image in enumerate(url_list):
                # Build the full path of the image file to save
                full_path = os.path.join(folder_path, f"{parse_name(str(row['gbifID']))}_{index}.jpg")

                # Download and save the image if it does not exist yet
                try:
                    if download_image(url_image, full_path):
                        process_image(
                            taxon_index,
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
                    warning(f"URL error: {row['identifier']}\nError: {e}")

                except Exception as e:
                    warning(f"Error: {e}")
                    pass
    except Exception as e:
        warning(f"Exception occurred in process row {e}")
    finally:
        semaphore_max_threads.release()
    return        

def process_image(
        taxon_index,
        full_path, row,
        counts_with_crops,
        counts_with_transformations_and_crops,
        training,
        initial_counts,
        model_to_discard: YOLO,
        model_to_crop: YOLO,
        semaphore_values,
        semaphore_models: threading.Semaphore):
    
    if is_corrupt_image(full_path):
        return
    
    not_discard = True
    if CHECK_IMAGES:
        with semaphore_models:
            not_discard = discard_bad_image(full_path, model_to_discard)
    
    if not_discard:
        number_of_transformations = TRANSFORMATIONS_PER_IMAGE
        total_transformations = None
        
        with semaphore_models:
            paths = crop_images(src_img=full_path, model_to_crop=model_to_crop, model_to_discard=model_to_discard) if MAX_NUM_OF_CROPS != 0 else [full_path]
        
        ## start critique section
        semaphore_values.acquire()
        try:
            not_processing = total_image_per_cat(taxon_index) - counts_with_transformations_and_crops[row[training]]
            if not_processing <= 0:
                cleanup_files(paths + [full_path])
                return
            
            initial_counts[row[training]] += 1
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
                    counts_with_crops[row[training]] += number_of_crops
                    counts_with_transformations_and_crops[row[training]] += number_of_crops + total_transformations
            else:
                counts_with_crops[row[training]] += number_of_crops
                counts_with_transformations_and_crops[row[training]] += total
        finally:
            semaphore_values.release()
        
        new_paths = [convert_to_webp(path) for path in paths]
        
        transformations = 0
        for k in range(number_of_transformations):
            if total_transformations and transformations >= total_transformations:
                break
            for new_path in new_paths:
                augment_image(new_path, k)
                transformations += 1
                if total_transformations and transformations >= total_transformations:
                    break
    else:
        pass

def cleanup_files(file_paths):
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except Exception:
            pass
            
def process_chunk(
        chunk, 
        taxon_index,
        training, 
        initial_counts, 
        counts_with_crops, 
        counts_with_transformations_and_crops,
        temp_image_path: str,
        model_to_discard: YOLO,
        model_to_crop: YOLO,
        key,
        semaphore_values=None):
    
    if semaphore_values is None:
        semaphore_values = threading.Semaphore()
        
    semaphore_models = threading.Semaphore()
    semaphore_max_threads = threading.Semaphore(MAX_THREADS_DOWNLOADING_PER_PROCESS)
    threads = []
    
    # Convert the chunk to a DataFrame only once
    chunk_df = pd.DataFrame(chunk)
    
    for _, row in chunk_df.iterrows():
        if USE_THREADS_TO_DOWNLOAD:
            thread = threading.Thread(target=process_row, 
                                      args=(
                                          taxon_index,
                                          row, 
                                          training, 
                                          initial_counts, 
                                          temp_image_path, 
                                          counts_with_crops, 
                                          counts_with_transformations_and_crops, 
                                          semaphore_values,
                                          semaphore_models,
                                          semaphore_max_threads,
                                          model_to_discard,
                                          model_to_crop))
            threads.append(thread)
            thread.start()
            try:
                semaphore_values.acquire()
                if counts_with_transformations_and_crops[key] >= total_image_per_cat(taxon_index):
                    break
            finally:
                semaphore_values.release()
        else:
            process_row(
                taxon_index,
                row, 
                training, 
                initial_counts, 
                temp_image_path, 
                counts_with_crops, 
                counts_with_transformations_and_crops, 
                semaphore_values,
                semaphore_models,
                semaphore_max_threads,
                model_to_discard,
                model_to_crop)
            
            if counts_with_transformations_and_crops[key] >= total_image_per_cat(taxon_index):
                break

    # Join threads if using multithreading
    if USE_THREADS_TO_DOWNLOAD:
        for thread in threads:
            thread.join()
            
    # Explicitly delete the DataFrame to free memory       
    del chunk_df
    
    return

def calculate_data_of_df(data_path: str, filter_colums=None, filters=None, print_bool=False):
    info('Calculating model counts')
    model_count_dict = {}
    chunksize = 10**3
    total_lines = 0

    # Convert filters to a dictionary for faster lookup
    filters_dict = {col: set(filt) for col, filt in zip(filter_colums, filters)}

    for column_name in TAXONOMIC_RANKS:
        unique_values = set()
        # Read CSV in chunks
        df_chunks = pd.read_csv(data_path, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
        
        for chunk in df_chunks:
            # Apply filters
            for col, filter_values in filters_dict.items():
                if TAXONOMIC_RANKS.index(col) <= TAXONOMIC_RANKS.index(column_name):
                    chunk = chunk[chunk[col].isin(filter_values)]
            
            # Update unique values
            unique_values.update(chunk[column_name].dropna().unique())
            
            # Increment total lines
            total_lines += len(chunk)
        
        model_count_dict[column_name] = len(unique_values)
        info(f"In the taxon of {column_name} we have a total of: {model_count_dict[column_name]}")
        if print_bool:
            info(f"There names of the different {column_name} are:\n{unique_values}\n")

    model_count_dict['total_lines'] = total_lines
    info(f"There are {total_lines} lines in this data frame")
    return model_count_dict

def initial_df_processing(df_occurrences, training, column_filters, filters, taxon_index):
    total_counts = {}  # Contains a dictionary of the taxon name with the number of existing data points.
    initial_counts = {}  # Contains a dictionary of the taxon name and keeps track of how many there are of each.
    dfs = []
    
    filter_dict = {col: set(filt) for col, filt in zip(column_filters, filters)}
    
    for chunk in df_occurrences:
        if training and filter_dict:
            for col_name, filter_values in filter_dict.items():
                if TAXONOMIC_RANKS.index(col_name) <= taxon_index:
                    chunk = chunk[chunk[col_name].isin(filter_values)]
                    
        value_counts = chunk[training].value_counts()
        
        for key, value in value_counts.items():
            if key not in total_counts:
                total_counts[key] = 0
                initial_counts[key] = 0
            total_counts[key] += value
        
        dfs.append(chunk)  # Add the chunk

    return initial_counts, total_counts, dfs

def no_more_images(counts: dict, taxon_index):
    limit = total_image_per_cat(taxon_index)
    return all(value >= limit for value in counts.values())

def increase_image_thread(image, total_images, valor, semaphore, key, taxon_index):
    with semaphore:
        if total_images.value > 0:
            while total_images.value < total_image_per_cat(taxon_index):
                total_images.value += 1
                semaphore.release()
                try:
                    augment_image(image, valor)
                finally:
                    semaphore.acquire()
        else:
            warning(f"No images found for class: {key}")
        
def increase_images(counts_with_transformations_and_crops, taxon_index):
    
    def increase_image_normal(image, total_images, valor):
        while total_images < total_image_per_cat(taxon_index):
            total_images += 1
            augment_image(image, valor)
    
    def return_n_images(images, n_images):
        return_images = []
        for image in images:
            return_images.append(image)
            if len(return_images) >= n_images:
                yield return_images
                return_images = []
        if return_images:
            yield return_images
                
    manager = Manager()
    
    for key, value in counts_with_transformations_and_crops.items():
        target_count = total_image_per_cat(taxon_index)
        if value < target_count:
            folder_path = os.path.join(TEMP_IMAGE_PATH, key)
            images = find_images(folder_path)
            random.shuffle(images)
            
            if USE_PROCESS_TO_AUMENT_IMG:
                total_images = manager.Value('i', len(images))
                semaphore = manager.Semaphore(1)
                
                with Pool(processes=NUMBER_OF_PROCESS) as pool_aument_images:
                    for images_chunk in return_n_images(images, NUMBER_OF_PROCESS * 2):
                        args = (total_images, value, semaphore, key)
                        args_aument = [(image, *args) for image in images_chunk]
                        pool_aument_images.starmap(increase_image_thread, args_aument)
                        with semaphore:
                            if total_images.value >= target_count:
                                break
            else:
                total_images = len(images)
                for image in images:
                    increase_image_normal(image, total_images, value)
                    if total_images >= target_count:
                        break

    return
                    
def filter_chunk(dfs, key, value, chunksize):
    filtered_chunks = []
    filtered_chunk_size = 0
    
    # Asegúrate de que el tamaño del chunk sea al menos 30
    if chunksize < 30:
        chunksize = 30
    
    for chunk in dfs:
        # Filtra el chunk basado en la condición dada
        filtered_chunk = chunk[chunk[key] == value]
        
        if not filtered_chunk.empty:
            filtered_chunks.append(filtered_chunk)
            filtered_chunk_size += len(filtered_chunk)
            
            # Si el tamaño del chunk filtrado supera el tamaño de salida deseado, yield el chunk concatenado
            if filtered_chunk_size >= chunksize:
                concatenated_chunk = pd.concat(filtered_chunks, ignore_index=True)
                
                yield concatenated_chunk
                
                # Resetear los chunks filtrados
                filtered_chunks = []
                filtered_chunk_size = 0
    
    # Yield el último chunk si no está vacío
    if filtered_chunks:
        concatenated_chunk = pd.concat(filtered_chunks, ignore_index=True)
        yield concatenated_chunk  
    
def filter_chunk_all(dfs, key, value, chunksize, size_chunk_list=NUMBER_OF_PROCESS):
    filtered_chunks = []
    filtered_chunk_size = 0
    concatenated_chunks = []

    # Asegúrate de que el tamaño del chunk sea al menos 30
    if chunksize < 30:
        chunksize = 30

    for chunk in dfs:
        # Filtra el chunk basado en la condición dada
        filtered_chunk = chunk[chunk[key] == value]

        if not filtered_chunk.empty:
            filtered_chunks.append(filtered_chunk)
            filtered_chunk_size += len(filtered_chunk)

            # Si el tamaño del chunk filtrado supera el tamaño de salida deseado, yield el chunk concatenado
            if filtered_chunk_size >= chunksize:
                concatenated_chunk = pd.concat(filtered_chunks, ignore_index=True)
                concatenated_chunks.append(concatenated_chunk)

                if len(concatenated_chunks) == size_chunk_list:
                    yield concatenated_chunks
                    concatenated_chunks = []

                # Resetear los chunks filtrados
                filtered_chunks = []
                filtered_chunk_size = 0

    # Yield el último chunk si no está vacío
    if filtered_chunks:
        concatenated_chunk = pd.concat(filtered_chunks, ignore_index=True)
        concatenated_chunks.append(concatenated_chunk)

    if concatenated_chunks:
        yield concatenated_chunks
        
def train_model(model, train_folder_path, model_name, start_time_func, execution_time_process_chunk, model_folder, taxon_index, counts_with_transformations_and_crops):
    start_time_train = time.time()
    train_bool = False
    min_value = 1
    min_cat = 2
    
    # Check if training should proceed
    if len(counts_with_transformations_and_crops) > 1:
        for value in counts_with_transformations_and_crops.values():
            if value > min_value:
                min_cat -= 1
                if min_cat == 0:
                    train_bool = True
                    break

    results = None
    if train_bool:
        # Train the model
        results = train_yolo_model(
            model=model,
            model_name=model_name,
            train_folder_path=train_folder_path,
            model_folder=model_folder,
            epochs=TRAIN_EPOCHS[taxon_index]
        )
        model_folder = os.path.dirname(results.save_dir) 
    else:
        # Ensure model folder exists
        os.makedirs(model_folder, exist_ok=True)
        
        path = os.path.join(model_folder, model_name)
        model_folder = path
        os.makedirs(path, exist_ok=True)
        
        # Write a file indicating only one category
        with open(os.path.join(path, 'data_path.txt'), 'w') as file:
            file.write('only one category')

    end_time_func = time.time()
    execution_time_func = end_time_func - start_time_func
    execution_time_train = end_time_func - start_time_train

    # Save execution time information
    save_information(model_folder, execution_time_func, execution_time_train, execution_time_process_chunk)
    
    return results, model_folder
    
def save_information(model_folder, execution_time_func, execution_time_train, execution_time_process_chunk):
    noti()
    info(f"The function took {execution_time_func} seconds to execute.")
    info(f"The training took {execution_time_train} seconds to execute.")
    info(f"The process of chunks took {execution_time_process_chunk} seconds to execute.")
    
    information_path = os.path.join(model_folder, 'information.txt')
    with open(information_path, 'w') as file:
        file.write(f"The function took {execution_time_func} seconds to execute.\n")
        file.write(f"The training took {execution_time_train} seconds to execute.\n")
        file.write(f"The process of chunks took {execution_time_process_chunk} seconds to execute.\n")
        
def copy_to_training_lines_process(folder, dest_path):
    folder_name = os.path.split(folder)[1]
    images = find_images(folder, extensions=['.webp', '.jpg'])
    copy_to_training_lines(dest_path, folder_name, images)
                
def train(
        model_folder: str = os.path.join(PATH_MODELS_TRAINED, "model_g"),
        path_model_to_train = MODEL_INIT,
        column_filters: list = [],
        filters: list = [],
        taxon_index: int = 0,
        temp_image_path = TEMP_IMAGE_PATH,
        train_folder_path = TRAINING_DEST_PATH,
        resume = False,
        download_images_bool = True,
        save_context = None,
        delete_previus_model = False,
        key = None
    ):

    start_time_func = time.time()

    if resume:
        save_context = SaveContext(load_state())
        if save_context.end_taxon is None or taxon_index < save_context.end_taxon:
            taxon_index = max(save_context.context_taxon_dict.keys())
            context_taxon = ContextTaxon(save_context.context_taxon_dict[taxon_index])
            model_folder = context_taxon.model_folder
            column_filters = context_taxon.column_filters
            filters = context_taxon.filters
            temp_image_path = context_taxon.temp_image_path
            train_folder_path = context_taxon.train_folder_path
            initial_counts = context_taxon.initial_counts
            total_counts = context_taxon.total_counts
            counts_with_crops = context_taxon.counts_with_crops
            counts_with_transformations_and_crops = context_taxon.counts_with_transformations_and_crops
            save_context.end_taxon = taxon_index

    if download_images_bool:
        empty_folder(temp_image_path)

    training = TAXONOMIC_RANKS[taxon_index]
    chunksize = 10**3
    df_occurrences = pd.read_csv(PROCESSED_DATA_CSV, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')

    start_time_proces_chunk = time.time()

    if download_images_bool:
        initial_counts, total_counts, dfs = initial_df_processing(df_occurrences, training, column_filters, filters, taxon_index)

        total_counts_correct = {key: value for key, value in total_counts.items() if value > (total_image_per_cat(taxon_index) * MIN_SAMPLE_PER_CATEGORY)}
        initial_counts = {key: 0 for key in total_counts_correct.keys()}
        total_counts = total_counts_correct

        if USE_PROCESS_TO_DOWNLOAD:
            manager = Manager()
            counts_with_crops = manager.dict(initial_counts)
            counts_with_transformations_and_crops = manager.dict(initial_counts)
            initial_counts = manager.dict(initial_counts)
            total_counts = manager.dict(total_counts)
            semaphore_values = manager.Semaphore(1)
        else:
            counts_with_crops = initial_counts.copy()
            counts_with_transformations_and_crops = initial_counts.copy()

        if SHUFFLE_DATAFRAME and taxon_index == 0:
            info('Shuffling dataframe')
            dfs = shuffle_DataFrame(dfs)

        info(f'Start processing chunks for model: {os.path.split(model_folder)[-1]}')

        model_to_discart = YOLO(DISCARD_MODEL_PATH)
        model_to_crop = YOLO(DETECT_MODEL_PATH)

        for key, _ in total_counts.items():
            if USE_PROCESS_TO_DOWNLOAD:
                with Pool(processes=NUMBER_OF_PROCESS) as pool_download:
                    for chunks in filter_chunk_all(dfs, training, key, chunksize):
                        args = (taxon_index, training, initial_counts, counts_with_crops, counts_with_transformations_and_crops, temp_image_path, model_to_discart, model_to_crop, key, semaphore_values)
                        chunks_with_args = [(chunk, *args) for chunk in chunks]
                        pool_download.starmap(process_chunk, chunks_with_args)
                        del chunks_with_args, chunks

                        if counts_with_transformations_and_crops[key] >= total_image_per_cat(taxon_index):
                            break
            else:
                for chunk in filter_chunk(dfs, training, key, chunksize):
                    process_chunk(chunk, taxon_index, training, initial_counts, counts_with_crops, counts_with_transformations_and_crops, temp_image_path, model_to_discart, model_to_crop, key)
                    if counts_with_transformations_and_crops[key] >= total_image_per_cat(taxon_index):
                        break

        del dfs
        info('Checking data images and increasing')
        increase_images(counts_with_transformations_and_crops, taxon_index)
        if USE_PROCESS_TO_DOWNLOAD:
            manager.shutdown()

    end_time_proces_chunk = time.time()

    model_name = os.path.split(model_folder)[-1]
    info(MODEL_INIT)
    chek_folder(model_folder)
    if download_images_bool:
        if delete_previus_model and os.path.exists(model_folder):
            empty_folder(model_folder)
            os.rmdir(model_folder)

        info(f"Using the following image counts\nDownloaded images: {initial_counts}\nCropped images: {counts_with_crops}\nTotal images: {counts_with_transformations_and_crops}")

    if download_images_bool:
        copy_to_training(temp_image_path, train_folder_path)
        training_yolo_bool = os.listdir(train_folder_path)
    else:
        training_yolo_bool = True

    if training_yolo_bool:
        info("Training: " + model_name)
        execution_time_process_chunk = end_time_proces_chunk - start_time_proces_chunk
        model = chek_model(path_model_to_train) or chek_model(MODEL_INIT)
        
        results, model_folder = train_model(
            model,
            train_folder_path,
            model_name,
            start_time_func,
            execution_time_process_chunk,
            model_folder,
            taxon_index,
            counts_with_transformations_and_crops
        )
        del model

        if TAXONOMIC_RANKS[taxon_index] == TAXONOMIC_RANKS[-1]:
            path = results.save_dir if results else train_folder_path
            with open(os.path.join(path, 'data.txt'), 'w') as file:
                for filter_item in filters:
                    for filter in filter_item:
                        file.write(filter + ",")
                file.write('especies: ')
                for specie in counts_with_transformations_and_crops.keys():
                    file.write(specie)

        path_to_model = os.path.join(results.save_dir, 'weights', 'best.pt') if results else path_model_to_train

        for key, _ in total_counts.items():
            if taxon_index < len(TAXONOMIC_RANKS) - 1:
                next_column_filters = column_filters.copy()
                next_filters = filters.copy()

                training_bool = True
                if TAXONOMIC_RANKS[taxon_index] not in next_column_filters:
                    next_column_filters.append(TAXONOMIC_RANKS[taxon_index])
                    next_filters.append([key])
                else:
                    index_filter = next_column_filters.index(TAXONOMIC_RANKS[taxon_index])
                    if key in next_filters[index_filter]:
                        next_filters[index_filter] = [key]
                    else:
                        training_bool = False

                if save_context is None:
                    save_context = SaveContext()
                context_taxon = ContextTaxon(
                    model_folder,
                    column_filters,
                    filters,
                    taxon_index,
                    temp_image_path,
                    train_folder_path,
                    initial_counts,
                    total_counts,
                    counts_with_crops,
                    counts_with_transformations_and_crops
                )
                save_context.add_context_taxon(context_taxon)
                save_state(save_context)

                if training_bool:
                    next_model_folder = os.path.join(model_folder, f"{TAXONOMIC_RANKS[taxon_index]}_{key}")
                    train(
                        model_folder=next_model_folder,
                        column_filters=next_column_filters,
                        filters=next_filters,
                        taxon_index=taxon_index + 1,
                        path_model_to_train=path_to_model,
                        resume=resume,
                        delete_previus_model=delete_previus_model
                    )
            else:
                info(f"Finished {column_filters} of {filters}")
    else:
        text = f"No data exists for these filters\n {column_filters}\n{filters}"
        chek_folder(model_folder)
        with open(os.path.join(model_folder, 'info.txt'), 'w') as file:
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
    model.train(data='training.yaml', epochs=TRAIN_EPOCHS, imgsz=IMAGE_SIZE, name=model_name)

def main():
    set_thread_priority()
    info('Starting training')

    if TRAIN_LOCAL:
        for path in training_data_path.values():
            empty_folder(path)

        directories = get_directories(IMAGES_FOLDER)
        train_folder(TAXONOMIC_RANKS[0][0], IMAGES_FOLDER)

        for taxon, index in TAXONOMIC_RANKS:
            for name in get_folders_by_level(IMAGES_FOLDER, max_level=index):
                directories = get_directories(IMAGES_FOLDER)
                name = os.path.basename(name)
                model_name = f"{taxon}_{name}"
                train_folder(model_name, name)
    else:
        filter_columns = [
            TAXONOMIC_RANKS[0],
            # TAXONOMIC_RANKS[1],
            # TAXONOMIC_RANKS[2],
            # TAXONOMIC_RANKS[3],
            # TAXONOMIC_RANKS[4]
        ]

        filters = [
            ['Gastropoda', 'Bivalvia', 'Cephalopoda', 'Polyplacophora'],
            # ['Seguenziida', 'Ellobiida', 'Runcinida', ...],
            # ['Helicarionidae', 'Euconulidae', 'Clausiliidae', ...],
            # ['Vidovicia', 'Cattania', 'Palaeotachea', ...],
            # ['Cornu Born, 1778', 'Cornu mazzullii (De Cristofori & Jan, 1832)', ...]
        ]

        taxon_index = 0

        train(
            column_filters=filter_columns,
            filters=filters,
            taxon_index=taxon_index,
            train_folder_path=TRAINING_DEST_PATH,
            download_images_bool=True,
            delete_previus_model=True,
        )

if __name__ == "__main__":
    main()

