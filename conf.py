import os
import random
from PIL import Image
from ultralytics import YOLO
import psutil
import time
from playsound import playsound

playsound('extras/noti.mp3')
VERBOSE = True  # If you want text to appear during executions

def warning(text: str):
    if VERBOSE:
        print(Colors.WARNING, text, Colors.ENDC)

def fail(text: str):
    print(Colors.FAIL, text, Colors.ENDC)

def info(text: str):
    if VERBOSE:
        print(Colors.OKGREEN, text, Colors.ENDC)

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
        warning(f"Discard model does not exist: {model}\nBad images will not be discarded")
        return None

def chek_folder(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder
    
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




SHUFFLE_DATAFRAME = False  # Indicates if the dataframe should be shuffled, useful for getting random data from the dataframe
TRAIN_LOCAL = False  # This means training will be done locally with the images already downloaded in the defined folder
TEMP_IMAGE_PATH = chek_folder("temp_images")  # This folder will temporarily store images before moving them to the training folder
DETECTION_IMAGE_PATH = chek_folder("detection_images")

PATH_MODELS_TRAINED = chek_folder(os.path.join('runs','classify'))
MODELS_FOLDER_PATH = chek_folder('yolo_models')
MODEL_INIT = os.path.join(MODELS_FOLDER_PATH,'yolov8n-cls.pt')  
model_init = chek_model(MODEL_INIT)


DISCARD_MODEL_PATH = os.path.join(MODELS_FOLDER_PATH,'yolo_discard.pt') 
DETECT_MODEL_PATH = os.path.join(MODELS_FOLDER_PATH,'yolo_detect.pt') 
    
PANDAS_CSV_PATH = chek_folder('pandas_files')
# This file contains the URLs with the already processed images.
PROCESSED_DATA_CSV = os.path.join(PANDAS_CSV_PATH,'parsed_occurrences_all.csv')

# This file contains the URLs with all the images, processed and unprocessed.
ALL_DATA_CSV = os.path.join(PANDAS_CSV_PATH,'parsed_occurrences_all.csv')  # All images regardless of quality

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

IMAGE_SIZE = 128
TRAIN_EPOCHS = 1

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
DATA_CONFIGURATIONS_YAML = chek_folder('train_configurations')
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
    ('class', 1),
    ('order', 2),
    ('family', 3),
    ('genus', 4)
    # ,('species',5)
]

# Training percentages for training, validation, and testing
VALIDATION_PERCENTAGE = 0.2
TESTING_PERCENTAGE = 0.2
TRAINING_PERCENTAGE = 1 - TESTING_PERCENTAGE - VALIDATION_PERCENTAGE
IMAGE_SAMPLE_COUNT = 3  # Maximum number of images per distinct class. Minimum should be 3.
if IMAGE_SAMPLE_COUNT < 3:
    fail(f"There must be at least 3 images per category")
    exit(-1)

TRANSFORMATIONS_PER_IMAGE = 2  # Specifies how many transformations each image should undergo.

CONF_TOP_5 = 0.9
PRIORITY = -10  # Sets the priority of processes in the system; requires running as sudo negative num more priority
