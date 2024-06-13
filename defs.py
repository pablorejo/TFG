import os
import platform
import random
import shutil  # To copy images to the training folder
import math
import pandas as pd
from conf import *

def parse_name(name):
    """Processes the string so it can be a valid file name.

    Args:
        name (str): The text string to be processed.

    Returns:
        str: The modified string.
    """
    replacements = {
        ":": "a",
        "<": "b",
        ">": "c",
        "\"": "e",
        "|": "f",
        "?": "g",
        "*": "h"
    }
    name = str(name)
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
        # If it is a directory, add to the list and return
        if os.path.isdir(path):
            return [path]
        else:
            return []
    
    # If we are not at the desired level, search in child directories
    if os.path.isdir(path):
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir():
                    # Build the path to the child directory
                    child_path = os.path.join(path, entry.name)
                    # Recursively call the function and accumulate the results
                    folders += get_folders_by_level(child_path, max_level, current_level + 1)
    
    return folders

def copy_to_training_file(data_type, file_path):
    """Copies images to the training folder.
    
    Args:
    file_path (str): The file containing the paths of the files to be copied, separated by \\n.
    data_type (str): The type of data, e.g., 'Bivalvia', 'Caudofaveata'.
    """
    with open(file_path) as file:
        lines = file.read().splitlines()
        random.shuffle(lines)  # This shuffles the list in-place
        copy_to_training_lines(dest_path=TRAINING_DEST_PATH,data_type=data_type, lines=lines)

def copy_to_training_detection(folder_path):
    """
    Copies the data from folder_path to the training folder along with their associated .txt files.
    """
    
    remove_images_without_txt(DETECTION_IMAGE_PATH)  # Remove images without detection if they exist in this directory
    empty_folder(TRAINING_DETECT_DEST_PATH)  # Empty the training image folder
    
    def copy_to_training(lines):
        """Copies images to the training folder.
        
        Args:
        lines (list): Array with the paths of the files to be copied.
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
            
    copy_to_training(find_images(folder_path, extensions=['.webp', '.jpg']))

def remove_images_without_txt(directory):
    """Removes all images in a directory and subdirectories that do not have a file with the same name ending in .txt."""
    
    # List all files in the directory
    files = os.listdir(directory)
    
    # Create a set of base names for .txt files
    txt_base_names = {os.path.splitext(file)[0] for file in files if file.endswith('.txt')}
    
    # Counter for the removed images
    removed_images = 0
    
    # Check each file in the directory
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):  # Add other image formats if necessary
            image_base_name = os.path.splitext(file)[0]
            
            # If the image base name is not in the set of .txt base names, remove the image
            if image_base_name not in txt_base_names:
                os.remove(os.path.join(directory, file))
                removed_images += 1
                info(f"Removed: {file}")
    
    info(f"Total removed images: {removed_images}")

def copy_to_training(folder_path,dest_path):
    """
    Copies the data from folder_path to the training folder.

    Args:
        folder_path (str): path that contains data
        dest_path (str): where you whant to move data to test valid and train

            Returns: true if there are images in the training folders
    """
    
    from conf import warning, info, fail
    empty_folder(dest_path)  # Empty the training image folder
    folders = get_folders_by_level(folder_path, 1)
    for folder in folders:
        copy_to_training_lines(dest_path,os.path.split(folder)[1], find_images(folder, extensions=['.webp', '.jpg']))
    
    if not os.listdir(dest_path):
        info("Empty folder")
        
    return os.listdir(dest_path)

def copy_to_training_lines(dest_path,data_type, lines, txt_associated=False):
    """Copies images to the training folder.
    
    Args:
    dest_path: path that go data tipes images 
    lines (list): Array with the paths of the files to be copied.
    data_type (str): The type of data, e.g., 'Bivalvia', 'Caudofaveata'.
    txt_associated (bool): Boolean that indicate if you want to copy also a file with same name that ends in txt.
    """
    from conf import TESTING_PERCENTAGE, TRAINING_PERCENTAGE, VALIDATION_PERCENTAGE,training_data_path, warning, info, VERBOSE
    if len(lines) >= 3:

        num_train = math.floor(len(lines) * TRAINING_PERCENTAGE)
        num_valid = math.floor(len(lines) * VALIDATION_PERCENTAGE)
        
        for i in range(num_train):
            copy_file(lines[i], os.path.join(dest_path,training_data_path["train"], data_type), txt_associated)
        
        for i in range(num_train, num_train + num_valid):
            copy_file(lines[i], os.path.join(dest_path,training_data_path["valid"], data_type), txt_associated)
            
        for i in range(num_train + num_valid, len(lines)):
            copy_file(lines[i], os.path.join(dest_path,training_data_path["test"], data_type), txt_associated)
    else:
        warning(f"Not enough images\nImages:\n")
        if VERBOSE:
            for line in lines:
                info(line + "\n")
            info("End of images\n\n")

def get_directories(folder_path):
    """Gets a list of directory names (not files) within the given folder_path."""
    directories = [name for name in os.listdir(folder_path)
                   if os.path.isdir(os.path.join(folder_path, name))]
    
    return directories

def find_images(directory, num_samples=None, extensions=['.webp']):
    """
    Gets all jpg images from a specific directory.
    
    Args:
    directory (str): Path to the directory from which to get all images.
    num_samples (int): If a value for num_samples is given, it will return that total number of images randomly.
    """
    jpg_images = []
    n = 0  # Total counter of jpg files found
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            for extension in extensions:
                if file.lower().endswith(extension):
                    n += 1
                    if num_samples is None:
                        jpg_images.append(os.path.join(root, file))
                    # If we have not yet collected num_samples images, simply add them
                    elif len(jpg_images) < num_samples:
                        jpg_images.append(os.path.join(root, file))
                    else:
                        # With probability num_samples/n, replace a random element
                        s = int(random.random() * n)
                        if s < num_samples:
                            jpg_images[s] = os.path.join(root, file)
    return jpg_images

def find_files(directory):
    # List to store the found files
    found_files = []

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

    return found_files

def shutdown_system():
    """Shuts down the system"""
    if platform.system() == "Windows":
        os.system('shutdown /s /t 1')
    elif platform.system() == "Linux":
        os.system('shutdown -h now')
    elif platform.system() == "Darwin":
        os.system('shutdown -h now')  
        
def empty_folder(folder_path):
    """Deletes everything in a folder"""
    if os.path.exists(folder_path):
        for name in os.listdir(folder_path):
            complete_path = os.path.join(folder_path, name)
            # Check if it is a file or a folder
            if os.path.isfile(complete_path) or os.path.islink(complete_path):
                os.remove(complete_path)  # Delete files or symbolic links
            elif os.path.isdir(complete_path):
                shutil.rmtree(complete_path)  # Delete subfolders and their content
    else: 
        os.mkdir(folder_path)

def get_GBIF(path: str):
    return os.path.split(path)[-1]

def copy_file(source_path, dest_path, txt_associated: bool = False):
    """Copies a  file to a folderand creates the folder if it doesn't exist"""
    if not os.path.exists(dest_path):
        os.makedirs(dest_path,exist_ok=True)
    shutil.copy(source_path, os.path.join(dest_path, get_GBIF(source_path)))
    
    if txt_associated:
        source_path = str(source_path).split(".")[0] + ".txt"
        shutil.copy(source_path, os.path.join(dest_path, get_GBIF(source_path)))

def shuffle_DataFrame(df: pd.DataFrame):
    """This function shuffles a dataframe to be random and saves it in a CSV designated in PROCESSED_DATA_CSV.

    Args:
        df (List DataFrame): List of DataFrames to be shuffled.
        
    return: Returns a new list of chunks with shuffled files.
    """
    from conf import PROCESSED_DATA_CSV, info, warning, fail
    # Define the output file name by adding 'shuffled_' to the original file name.
    
    # Initialize an empty list to store the shuffled chunks.
    chunks_list = []
    
    # Shuffle each chunk and add it to the list of chunks.
    for chunk in df:
        shuffled_chunk = chunk.sample(frac=1).reset_index(drop=True)
        chunks_list.append(shuffled_chunk)
    
    # Concatenate all shuffled chunks into a single DataFrame.
    shuffled_df = pd.concat(chunks_list, ignore_index=True)
    
    # Save the shuffled DataFrame in a new CSV file.
    shuffled_df.to_csv(PROCESSED_DATA_CSV, index=False)
    
    info(f"Shuffled file saved as: {PROCESSED_DATA_CSV}")
    
    return chunks_list
