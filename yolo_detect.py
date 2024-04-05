from ultralytics import YOLO,engine,engine
import cv2
# Carga un modelo preentrenado
# model = YOLO('yolov8n.pt')
# results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
# metrics = model.val()

def recortarImagenes(src_img: str, model: YOLO):
    """
        Esta función recorta las imagenes basandose en un modelo entrenado que le devuelve las secciones.
    """
    image = cv2.imread(src_img)
    results = model(image)
    numero = 0
    for result in results:
        boxes = result.boxes
        
        # Extraer las coordenadas del cuadro delimitador, convertidas a enteros
        x_min, y_min, x_max, y_max = boxes.numpy().xyxy[0][:4]
        
        # Recortar la imagen
        cropped_img = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        # Guardar la imagen recortada
        cropped_img_path = f'{src_img}_cropped_{numero}.jpg'
        numero +=1
        cv2.imwrite(cropped_img_path, cropped_img)
        print(f'Imagen recortada guardada en: {cropped_img_path}')
        
if __name__ == "__main__":
    recortarImagenes(src_img="prueba.png", model=YOLO('yolov8n.pt'))