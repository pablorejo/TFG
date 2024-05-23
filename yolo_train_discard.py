from ultralytics import YOLO
from conf import TRAINING_DEST_PATH, GOOD_IMAGE_FILE, BAD_IMAGE_FILE, IMAGE_SIZE
from defs import empty_folder, copy_to_training_file

empty_folder(TRAINING_DEST_PATH)
copy_to_training_file('good', GOOD_IMAGE_FILE)
copy_to_training_file('bad', BAD_IMAGE_FILE)

model = YOLO('yolov8n-cls.pt')
results = model.train(epochs=3, imgsz=IMAGE_SIZE)
