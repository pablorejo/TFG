import os
import random
from PIL import Image
from ultralytics import YOLO
import psutil
import time
from playsound import playsound
import torch


VERBOSE = True  # If you want text to appear during executions
# Example of ANSI escape sequences for different colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m WARNING: '
    FAIL = '\033[91m ERROR: '
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def warning(text: str):
    if VERBOSE:
        print(Colors.WARNING, text, Colors.ENDC)

def fail(text: str):
    print(Colors.FAIL, text, Colors.ENDC)

def info(text: str):
    if VERBOSE:
        print(Colors.OKGREEN, text, Colors.ENDC)

def noti():
    # Asegurarse de que la ruta al archivo de sonido es correcta
    sound_file = os.path.join('extras', 'noti.mp3')
    
    # Verificar si el archivo de sonido existe
    if os.path.exists(sound_file):
        try:
            playsound(sound_file)
            info('Notification sound')
        except UnicodeDecodeError as e:
            fail(f"Error UnicodeDecodeError playing sound: {e}")
        
        except Exception as e:
            fail(f"Error playing sound: {e}")
    else:
        warning(f"Sound file not found: {sound_file}")
noti()

def set_thread_priority():
    p = psutil.Process(os.getpid())
    try:
        p.nice(PRIORITY)
        if (PRIORITY > 0):
            info(f"Thread priority set to LOW")
        else:
            info(f"Thread priority set to HIGHT")

    except psutil.AccessDenied as e:
        fail(f"You need to run as superuser to set priority correctly")
        time.sleep(2)
    except Exception as e:
        fail(f"Failed to set thread priority: {e}")
    
def chek_model(model: str):
    if os.path.exists(model): 
        return YOLO(model)
    else:
        warning(f"Model does not exist: {model}")
        return None

def chek_folder(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

SHUFFLE_DATAFRAME = False  # Indicates if the dataframe should be shuffled, useful for getting random data from the dataframe
TRAIN_LOCAL = False  # This means training will be done locally with the images already downloaded in the defined folder
TEMP_IMAGE_PATH = chek_folder("temp_images")  # This folder will temporarily store images before moving them to the training folder
DETECTION_IMAGE_PATH = chek_folder("detection_images")
DISCARD_IMAGE_PATH = chek_folder("discard_images")
PATH_MODELS_TRAINED = chek_folder(os.path.join('runs','classify'))
MODELS_FOLDER_PATH = chek_folder('yolo_models')
MODEL_INIT = os.path.join(MODELS_FOLDER_PATH,'yolo_init.pt') # this is a training model in class rank  

NAME_MODEL_DISCARD = 'yolo_discard'
DISCARD_MODEL_PATH = os.path.join(MODELS_FOLDER_PATH,f'{NAME_MODEL_DISCARD}.pt') 

NAME_MODEL_DETECT = 'yolo_detect'
DETECT_MODEL_PATH = os.path.join(MODELS_FOLDER_PATH,f'{NAME_MODEL_DETECT}.pt') 
    
PANDAS_CSV_PATH = chek_folder('pandas_files')
# This file contains the URLs with the already processed images.
PROCESSED_DATA_CSV = os.path.join(PANDAS_CSV_PATH,'parsed_occurrences.csv')

# This file contains the URLs with all the images, processed and unprocessed.
ALL_DATA_CSV = os.path.join(PANDAS_CSV_PATH,'parsed_occurrences.csv')  # All images regardless of quality

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DATA_CSV):
        fail(f"File {PROCESSED_DATA_CSV} does not exist. A CSV file with the data is required.")
        print("""What do you want to do?
    1) Use the all data CSV file as the processed file
    2) Cancel
    3) Continue anyway""")
        number = int(input(": "))
        if os.path.exists(ALL_DATA_CSV):
            if number == 1:
                PROCESSED_DATA_CSV = ALL_DATA_CSV
            elif number == 2:
                exit(0)
            else:
                pass
        else:
            fail(f"File {ALL_DATA_CSV} does not exist. A CSV file with the data is required to execute yolo_train_img.\n")

Image.MAX_IMAGE_PIXELS = None  # Allow unlimited image size

# DEVICE Specifies the computational device(s) options:
## 'cpu': train in cpu
## 'mps': train in MPS for Apple silicon
## 0,1: multiple GPUs
## 0: single GPU
# DEVICE = 0
DEVICE = 'cuda:0' if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

# Path to the folder containing all images
IMAGES_FOLDER = chek_folder('images')

# Types of images in the first part of the training, good and bad
DISCARD_TXT_PATH = chek_folder("txt_files")
GOOD_IMAGE_FILE = os.path.join(DISCARD_TXT_PATH,'good_images.txt')
BAD_IMAGE_FILE = os.path.join(DISCARD_TXT_PATH,'bad_images.txt')
types = {
    'good': 'buenas',
    'bad': 'malas'
}

IMAGE_DATA = os.path.join(DISCARD_TXT_PATH, 'multimedia.txt')
OCCURRENCE_DATA = os.path.join(DISCARD_TXT_PATH, 'occurrence.txt')


# Path where the training data will be stored
TRAINING_DEST_PATH = chek_folder(os.path.join('datasets','imagenet10'))
training_data_path = {
    'train': 'train',
    'test': 'test',
    'valid': 'val',
}

TRAINING_DETECT_DEST_PATH = chek_folder(os.path.join('datasets','detect'))
training_detect_data_path = {
    'train': os.path.join(TRAINING_DETECT_DEST_PATH,'train'),
    'test': os.path.join(TRAINING_DETECT_DEST_PATH,'test'),
    'valid': os.path.join(TRAINING_DETECT_DEST_PATH,'valid'),
}

# Taxonomic ranks and their folder recursion levels for recursive training
TAXONOMIC_RANKS = [
    'class',
    'order',
    'family',
    'genus',
    'acceptedScientificName'
]



# Training percentages for training, validation, and testing
VALIDATION_PERCENTAGE = 0.1
TESTING_PERCENTAGE = 0.05
TRAINING_PERCENTAGE = 1 - TESTING_PERCENTAGE - VALIDATION_PERCENTAGE
MAX_NUM_OF_CROPS = 20 # In a image only crop 5 times
# IMAGE_SAMPLE_COUNT = [
#     5000,
#     4000,
#     3000,
#     2000,
#     600
# ]
MIN_SAMPLE_PER_CATEGORY = 0.3 # 30% of the total image per cat.
IMAGE_SAMPLE_COUNT = [
    1,
    1,
    1,
    1,
    1
]
CHECK_IMAGES = False
TRANSFORMATIONS_PER_IMAGE = 3  # Specifies how many transformations each image should undergo.
def total_image_per_cat(taxon_index):
    return IMAGE_SAMPLE_COUNT[taxon_index] + IMAGE_SAMPLE_COUNT[taxon_index] * TRANSFORMATIONS_PER_IMAGE

NUM_WORKERS = 8 # If use cuda for train number of process usees per gpu
BATCH = 4 # Number of bathsize, increase ram use.
BATCH = 16 # Number of bathsize, increase ram use.

# IMAGE_SIZE = 1024
IMAGE_SIZE = 128
# TRAIN_EPOCHS = [
#     45,
#     45,
#     45,
#     45,
#     45
# ]
TRAIN_EPOCHS = [
    1,
    1,
    1,
    1,
    1
]

CONF_TOP_5 = 0.9 # Conf to discard model if conf is less than this de image is bad.
PRIORITY = -10  # Sets the priority of processes in the system; requires running as sudo negative num more priority
USE_THREADS_TO_DOWNLOAD = True # If you want download data with threads (quickly) set it True if you want sequence set it False
MAX_THREADS_DOWNLOADING_PER_PROCESS = 10
USE_PROCESS_TO_DOWNLOAD = False
NUMBER_OF_PROCESS = 8
USE_PROCESS_TO_AUMENT_IMG = False

NUMBER_OF_PROCESS_PANDAS = 14

USE_PROCESS_TO_COPI_IMG = False
NUMBER_OF_PROCESS_TO_COPY = 14