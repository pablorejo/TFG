
import os, sys
from ultralytics import YOLO
from globales import *
from funciones import recortar_imagenes

class Animal:
    
    def __init__(self,rangos_taxonomicos: dict):
        self.clase = rangos_taxonomicos['clase']
        self.orden = rangos_taxonomicos['orden']
        self.familia = rangos_taxonomicos['familia']
        self.genero = rangos_taxonomicos['genero']
        self.especie = rangos_taxonomicos['especie']
        
    def to_string(self):
        cadena = f"clase: {self.clase}"
        cadena = cadena + f"\norden: {self.orden}"
        cadena = cadena + f"\nfamilia: {self.familia}"
        cadena = cadena + f"\ngenero: {self.genero}"
        cadena = cadena + f"\nespecie: {self.especie}"
        return cadena
    
    def poner_imagen(self, img_ruta):
        self.img_ruta = img_ruta
        
    def set_rastro(self, rastro):
        self.rastro = rastro


def predecir_basico(imgPAHT: str, model: YOLO = YOLO('runs/classify/train2/weights/best.pt' )):
        # Realiza la predicción en la imagen
        results = model(imgPAHT)

        # Suponiendo que `results` es una lista de objetos `Results`
        for result in results:
            probs = result.probs  # Probs object for classification outputs
            # Obtiene el índice de la clase top 1 (la más probable)
            top5_str = []
            for top in probs.top5:
                top5_str.append(result.names[top])
            
            return top5_str, probs.top1conf
            
def predecir_imagen(img_path: str):
    """Esta funcion nos devuelve una clase imagen que contiene las caracteristicas de este animal
    Args:
        imgPath (str): ruta a la imagen a predecir
    """

    rangos_taxonomicos = [
        'clase',
        'orden',
        'familia',
        'genero',
        'especie',
    ]
    
    taxones = {}
    top5_resultado_bool = False
    top5_resultado = []
    nombre_modelo = ""
    rastro =  []
    for taxon in rangos_taxonomicos:
        str_modelo = f'runs/classify/modelo_entrenando_{nombre_modelo}/weights/best.pt'
        model = YOLO(str_modelo)
        top5,conf = predecir_basico(imgPAHT=img_path,model=model)
        if conf < CONF_TOP_5 and not top5_resultado_bool:
            top5_resultado_bool = True
            top5_resultado = top5
        
        rastro.append(conf)
        nombre_modelo = "_" + top5[0]
        taxones[taxon] = top5[0]
    
    animal = Animal(rangos_taxonomicos)
    animal.set_rastro(rastro)
    
    info(animal.toString())
    return animal

def predecir_imagenes(imagenes):
    
    
    for imagen in imagenes:
        animal = Animal(predecir_imagen(imagen))
        animal.rastro
    
    pass

def predecir(args:str):
    
    imagenes = args.split(',')
    animales = []
    if len(imagenes) == 1:
        imagenes_cropped = recortar_imagenes(imagenes[0],delete_original=False)
        for imagen in imagenes_cropped:
            animal = Animal(predecir_imagen(imagen))
            animal.poner_imagen(imagen)
    elif len(imagenes) > 1:
        predecir_imagenes(imagenes)
    else:
        fail("No se han proporcionado imagenes")
        exit(0)
        
        
        

if __name__ == "__main__":
    if (len(sys.argv) <= 1):
        fail("Falta argumento de la imagen\npython3 yolo_predict.py 'img_path'")
        exit(-1)
    
    else:
        predecir('prueba.webp')
        # predecir(sys.argv[1])