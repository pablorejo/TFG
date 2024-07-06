from conf import (
    info,
    warning,
    fail,
    TRAINING_DEST_PATH,
    DETECTION_IMAGE_PATH,
    TRAINING_DETECT_DEST_PATH,
    VALIDATION_PERCENTAGE,
    TESTING_PERCENTAGE,
    training_detect_data_path,
    TRAINING_PERCENTAGE,
    USE_PROCESS_TO_COPI_IMG,
    NUMBER_OF_PROCESS_TO_COPY,
    training_data_path,
)
import os
import random
import shutil  # To copy images to the training folder
import math
import pandas as pd
from multiprocessing import Pool
from threading import Thread,Semaphore

def parse_name(name):
    """
    Processes the string so it can be a valid file name.

    Args:
        name (str): The text string to be processed.

    Returns:
        str: The modified string.
    """
    # Diccionario de caracteres a reemplazar
    replacements = {
        ":": "a",
        "<": "b",
        ">": "c",
        "\"": "e",
        "|": "f",
        "?": "g",
        "*": "h",
        "/": "_",  # Añadiendo barra para Unix
        "\\": "_"  # Añadiendo barra para Windows
    }

    # Asegurarse de que el nombre sea una cadena
    name = str(name)
    
    # Reemplazar caracteres no válidos
    for search, replace in replacements.items():
        name = name.replace(search, replace)
    
    return name

def get_folders_by_level(path, max_level, current_level=0):
    """
    Returns a list of folders at a certain recursion level.

    Args:
    path (str): The starting path.
    max_level (int): The specific recursion level to search for.
    current_level (int): The current recursion level (used internally).

    Returns:
    list: A list with relative paths to folders at the specified level.
    """
    folders = []

    # Check if we have reached the desired level
    if current_level == max_level:
        if os.path.isdir(path):
            return [path]
        return []

    # If we are not at the desired level, search in child directories
    if os.path.isdir(path):
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir():
                    child_path = os.path.join(path, entry.name)
                    # Recursively call the function and accumulate the results
                    folders.extend(get_folders_by_level(child_path, max_level, current_level + 1))
    
    return folders

def copy_to_training_file(data_type, file_path):
    """
    Copies images to the training folder.

    Args:
    data_type (str): The type of data, e.g., 'Bivalvia', 'Caudofaveata'.
    file_path (str): The file containing the paths of the files to be copied, separated by \\n.
    """
    try:
        with open(file_path) as file:
            lines = file.read().splitlines()
            random.shuffle(lines)  # This shuffles the list in-place
            copy_to_training_lines(dest_path=TRAINING_DEST_PATH, data_type=data_type, lines=lines)
    except FileNotFoundError:
        fail(f"File not found: {file_path}")
    except Exception as e:
        fail(f"An error occurred while processing {file_path}: {e}")

def copy_to_training_detection(folder_path):
    """
    Copies the data from folder_path to the training folder along with their associated .txt files.
    """
    remove_images_without_txt(DETECTION_IMAGE_PATH)  # Remove images without detection if they exist in this directory
    empty_folder(TRAINING_DETECT_DEST_PATH)  # Empty the training image folder
    
    def copy_to_training(lines):
        """
        Copies images to the training folder.

        Args:
        lines (list): List with the paths of the files to be copied.
        """
        if len(lines) >= 3:
            num_valid = math.ceil(len(lines) * VALIDATION_PERCENTAGE)
            num_test = math.ceil(len(lines) * TESTING_PERCENTAGE)
            
            for i in range(num_valid):
                copy_file(lines[i], os.path.join(training_detect_data_path["valid"]), txt_associated=True)
            
            for i in range(num_valid, num_valid + num_test):
                copy_file(lines[i], os.path.join(training_detect_data_path["test"]), txt_associated=True)
                
            for i in range(num_valid + num_test, len(lines)):
                copy_file(lines[i], os.path.join(training_detect_data_path["train"]), txt_associated=True)
        else:
            warning(f"Not enough images\nImages:\n")

    # Find and copy images
    images = find_images(folder_path, extensions=['.webp', '.jpg'])
    copy_to_training(images)

def remove_images_without_txt(directory):
    """
    Removes all images in a directory and subdirectories that do not have a file with the same name ending in .txt.
    
    Args:
    directory (str): The directory to check for images without associated .txt files.
    """
    removed_images = 0
    
    for root, _, files in os.walk(directory):
        # Create a set of base names for .txt files in the current directory
        txt_base_names = {os.path.splitext(file)[0] for file in files if file.endswith('.txt')}
        
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):  # Add other image formats if necessary
                image_base_name = os.path.splitext(file)[0]
                
                # If the image base name is not in the set of .txt base names, remove the image
                if image_base_name not in txt_base_names:
                    try:
                        os.remove(os.path.join(root, file))
                        removed_images += 1
                        info(f"Removed: {os.path.join(root, file)}")
                    except Exception as e:
                        warning(f"Failed to remove {os.path.join(root, file)}: {e}")
    
    info(f"Total removed images: {removed_images}")

def copy_to_training(folder_path, dest_path):
    """
    Copies the data from folder_path to the training folder.

    Args:
        folder_path (str): Path that contains the data.
        dest_path (str): Destination path where data will be moved for training, validation, and testing.

    Returns:
        bool: True if there are images in the training folders, False otherwise.
    """
    info('Copying images to training folder')
    empty_folder(dest_path)  # Empty the training image folder
    folders = get_folders_by_level(folder_path, 1)
    
    for folder in folders:
        info(f"Copying images from folder: {folder}")
        images = find_images(folder, extensions=['.webp', '.jpg'])
        if images:
            copy_to_training_lines(dest_path, os.path.basename(folder), images)
    
    if not os.listdir(dest_path):
        info("Training folder is empty")
        return False
        
    return True

def copy_to_training_lines(dest_path, data_type, lines, txt_associated=False):
    """
    Copies images to the training folder.

    Args:
    dest_path (str): Path where the data type images will go.
    data_type (str): The type of data, e.g., 'Bivalvia', 'Caudofaveata'.
    lines (list): List of file paths to be copied.
    txt_associated (bool): Whether to copy associated .txt files. Defaults to False.
    """
    if len(lines) < 3:
        warning("Not enough images to copy")
        return

    num_train = math.floor(len(lines) * TRAINING_PERCENTAGE)
    num_valid = math.floor(len(lines) * VALIDATION_PERCENTAGE)

    def process_lines(segment, folder):
        if USE_PROCESS_TO_COPI_IMG:
            args = [
                os.path.join(dest_path, folder, data_type),
                txt_associated
            ]
            args_folders = [(line, *args) for line in segment]
            with Pool(processes=NUMBER_OF_PROCESS_TO_COPY) as copy_pool:
                copy_pool.starmap(copy_file, args_folders)
        else:
            for line in segment:
                dest = os.path.join(dest_path, folder, data_type)
                copy_file(line, dest, txt_associated)

    # Copy training data
    info(f'Copying data type: {data_type} to train')
    process_lines(lines[:num_train], training_data_path['train'])

    # Copy validation data
    info(f'Copying data type: {data_type} to valid')
    process_lines(lines[num_train:num_train + num_valid], training_data_path['valid'])

    # Copy test data
    info(f'Copying data type: {data_type} to test')
    process_lines(lines[num_train + num_valid:], training_data_path['test'])

def get_directories(folder_path):
    """
    Gets a list of directory names (not files) within the given folder_path.

    Args:
    folder_path (str): The path to the folder to list directories from.

    Returns:
    list: A list of directory names within the given folder_path.
    """
    try:
        directories = [name for name in os.listdir(folder_path)
                       if os.path.isdir(os.path.join(folder_path, name))]
        return directories
    except FileNotFoundError:
        warning(f"Folder not found: {folder_path}")
        return []
    except PermissionError:
        warning(f"Permission denied: {folder_path}")
        return []
    except Exception as e:
        fail(f"An error occurred while listing directories in {folder_path}: {e}")
        return []

def find_images(directory, num_samples=None, extensions=['.webp']):
    """
    Gets all images with specified extensions from a specific directory.
    
    Args:
    directory (str): Path to the directory from which to get all images.
    num_samples (int): If a value for num_samples is given, it will return that total number of images randomly.
    extensions (list): List of file extensions to look for. Defaults to ['.webp'].
    
    Returns:
    list: List of image file paths.
    """
    images = []
    n = 0  # Total counter of files found with the specified extensions
    
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    n += 1
                    file_path = os.path.join(root, file)
                    if num_samples is None:
                        images.append(file_path)
                    elif len(images) < num_samples:
                        images.append(file_path)
                    else:
                        s = int(random.random() * n)
                        if s < num_samples:
                            images[s] = file_path

        return images

    except Exception as e:
        fail(f"An error occurred while finding images in {directory}: {e}")
        return []

def find_files(directory):
    """
    Finds all .txt and .csv files in the specified directory.

    Args:
    directory (str): The path to the directory to search for files.

    Returns:
    list: A list of file paths for all .txt and .csv files found in the directory.
    """
    found_files = []

    try:
        # Iterate through all files and directories in the specified directory
        for file in os.listdir(directory):
            # Build the complete file path
            complete_path = os.path.join(directory, file)
            
            # Check if it is a file
            if os.path.isfile(complete_path):
                # Get the file extension
                _, extension = os.path.splitext(file)
                
                # Check if the extension is .txt or .csv
                if extension in ['.txt', '.csv']:
                    found_files.append(complete_path)
                    
    except FileNotFoundError:
        warning(f"Directory not found: {directory}")
    except PermissionError:
        warning(f"Permission denied: {directory}")
    except Exception as e:
        fail(f"An error occurred while listing files in {directory}: {e}")

    return found_files
    
def empty_folder(folder_path):
    """
    Deletes everything in a folder. If the folder does not exist, it will be created.

    Args:
    folder_path (str): The path to the folder to be emptied.
    """
    if os.path.exists(folder_path):
        for name in os.listdir(folder_path):
            complete_path = os.path.join(folder_path, name)
            try:
                if os.path.isfile(complete_path) or os.path.islink(complete_path):
                    os.remove(complete_path)  # Delete files or symbolic links
                    info(f"Deleted file: {complete_path}")
                elif os.path.isdir(complete_path):
                    shutil.rmtree(complete_path)  # Delete subfolders and their content
                    info(f"Deleted directory: {complete_path}")
            except Exception as e:
                warning(f"Failed to delete {complete_path}. Reason: {e}")
    else: 
        try:
            os.mkdir(folder_path)
            info(f"Created directory: {folder_path}")
        except Exception as e:
            fail(f"Failed to create directory {folder_path}. Reason: {e}")

def get_GBIF(path: str):
    return os.path.split(path)[-1]

def copy_file(source_path, dest_path, txt_associated: bool = False):
    """
    Copies a file to a folder and creates the folder if it doesn't exist.
    
    Args:
    source_path (str): The path of the source file to copy.
    dest_path (str): The destination directory where the file will be copied.
    txt_associated (bool): Whether to copy an associated .txt file with the same base name. Defaults to False.
    """
    try:
        # Ensure the destination directory exists
        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)

        # Copy the main file
        dest_file_path = os.path.join(dest_path, get_GBIF(source_path))
        shutil.copy(source_path, dest_file_path)
        info(f"Copied file: {source_path} to {dest_file_path}")

        # Copy the associated .txt file if specified
        if txt_associated:
            txt_source_path = os.path.splitext(source_path)[0] + ".txt"
            if os.path.exists(txt_source_path):
                txt_dest_file_path = os.path.join(dest_path, get_GBIF(txt_source_path))
                shutil.copy(txt_source_path, txt_dest_file_path)
                info(f"Copied associated txt file: {txt_source_path} to {txt_dest_file_path}")
            else:
                warning(f"Associated txt file not found: {txt_source_path}")

    except Exception as e:
        fail(f"Failed to copy file: {source_path} to {dest_path}. Reason: {e}")

def shuffle_DataFrame(df_list: list, output_csv: str):
    """
    Shuffles a list of DataFrames to be random and saves it in a designated CSV.

    Args:
        df_list (list): List of DataFrames to be shuffled.
        output_csv (str): Path to the output CSV file where the shuffled DataFrame will be saved.
    
    Returns:
        list: A new list of chunks with shuffled DataFrames.
    """
    try:
        # Initialize an empty list to store the shuffled chunks
        chunks_list = []
        
        # Shuffle each chunk and add it to the list of chunks
        for chunk in df_list:
            shuffled_chunk = chunk.sample(frac=1).reset_index(drop=True)
            chunks_list.append(shuffled_chunk)
        
        # Concatenate all shuffled chunks into a single DataFrame
        shuffled_df = pd.concat(chunks_list, ignore_index=True)
        
        # Save the shuffled DataFrame in a new CSV file
        shuffled_df.to_csv(output_csv, index=False)
        
        info(f"Shuffled file saved as: {output_csv}")
        return chunks_list

    except Exception as e:
        fail(f"Failed to shuffle and save DataFrame. Reason: {e}")
        return []
