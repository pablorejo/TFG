
import os
from ultralytics import YOLO

class Animal:
    
    def __init__(self, clase, orden, familia, genero, especie,rangos_taxonomicos: dict):
        if (rangos_taxonomicos == None):
            self.clase = clase
            self.orden = orden
            self.familia = familia
            self.genero = genero
            self.especie = especie
        else:
            self.clase = rangos_taxonomicos['clase']
            self.orden = rangos_taxonomicos['orden']
            self.familia = rangos_taxonomicos['familia']
            self.genero = rangos_taxonomicos['genero']
            self.especie = rangos_taxonomicos['especie']
        
    def toString(self):
        cadena = f"clase: {self.clase}"
        cadena = cadena + f"\norden: {self.orden}"
        cadena = cadena + f"\nfamilia: {self.familia}"
        cadena = cadena + f"\ngenero: {self.genero}"
        cadena = cadena + f"\nespecie: {self.especie}"
        return cadena
    
        
def predecir(imgPath):
    """Esta funcion nos devuelve una clase imagen que contiene las caracteristicas de este animal

    Args:
        imgPath (_type_): _description_
    """

    def predecirBasico(imgPAHT: str, model: YOLO = YOLO('../runs/classify/train2/weights/best.pt' )):
        # Realiza la predicción en la imagen
        results = model(imgPAHT)

        # Suponiendo que `results` es una lista de objetos `Results`
        for result in results:
            probs = result.probs  # Probs object for classification outputs
            # Obtiene el índice de la clase top 1 (la más probable)
            top1_class_index = probs.top1
            return top1_class_index
        

    rangos_taxonomicos = [
        'clase',
        'orden',
        'familia',
        'genero',
        'especie',
    ]
    
    animal = Animal()
    taxones = {}
    for taxon in rangos_taxonomicos:
        str_modelo = f'/runs/classify/modelo_entrenando_{taxon}/weights/best.pt'
        model = YOLO(str_modelo)
        predicho = predecirBasico(imgPAHT=imgPath,model=model)
        taxones[taxon] = predicho
    
    animal = Animal(rangos_taxonomicos)
    print(animal.toString())
    return animal