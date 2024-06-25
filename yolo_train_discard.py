from ultralytics import YOLO
from conf import *
from defs import empty_folder, copy_to_training_file
from defs_img import train_yolo_model
from os import path

def main():
    empty_folder(TRAINING_DEST_PATH)
    copy_to_training_file(types['good'], GOOD_IMAGE_FILE)
    copy_to_training_file(types['bad'], BAD_IMAGE_FILE)
    model = chek_model(MODEL_INIT)
    model_folder = chek_folder(path.join(PATH_MODELS_TRAINED,NAME_MODEL_DISCARD))
    info(model_folder)
    train_yolo_model(model,model_name=NAME_MODEL_DISCARD,train_folder_path=TRAINING_DEST_PATH,model_folder=model_folder, epochs=30)
    
if __name__ == "__main__":
    main()
