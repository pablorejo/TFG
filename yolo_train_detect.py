from ultralytics import YOLO
from defs import copy_to_training_detection
from conf import IMAGE_SIZE, DETECTION_IMAGE_PATH,TRAIN_EPOCHS
import os

def process_txt_data(directory):
    """
    Processes the detection data in the folder so that only one class exists.

    Args:
    directory (str): The path to the directory where the .txt files are located.
    """
    # Iterate through the files in the specified directory
    for file in os.listdir(directory):
        # Check if the file ends with .txt
        if file.endswith('.txt'):
            # Add the full path of the file to the list
            path = os.path.join(directory, file)
            with open(path, 'r') as file_read:
                lines = file_read.readlines()
                new_lines = []
                for line in lines:
                    if str(line).split(' ', maxsplit=1)[0] != '0':
                        print(f"The line {line} contains a one, the file is: {path}")
                    new_line = '0 ' + str(line).split(' ', maxsplit=1)[1]
                    new_lines.append(new_line)
            
            file_read.close()
            with open(path, 'w') as file_write:
                file_write.writelines(new_lines)
                file_write.close()


def main():
    copy_to_training_detection(DETECTION_IMAGE_PATH)  # Copy the data to the training path
    process_txt_data(DETECTION_IMAGE_PATH)  # Process the data so that only one class exists, in case the user made a mistake
    model = YOLO('yolov8n.pt')  # Get the model for classification
    results = model.train(data='conf_detect.yaml', epochs=TRAIN_EPOCHS, imgsz=IMAGE_SIZE)  # Train it with 30 epochs and the conf_detect.yaml configuration file

if __name__ == "__main__":
    main()