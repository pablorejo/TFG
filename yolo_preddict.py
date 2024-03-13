from ultralytics import YOLO
import globales
import random
if __name__ == "__main__":
    # Load a model
    model = YOLO('runs/classify/train2/weights/best.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    imagenes = globales.encontrar_imagenes_jpg('imagenes',num_muestras=10)
    random.shuffle(imagenes)
    imagenes = imagenes[1:10]
    results = model(imagenes)  # return a list of Results objects
    
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen
        # result.save(filename='result.jpg')  # save to disk