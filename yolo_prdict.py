import sys
from conf import *
class Taxon:
    def __init__(self, name_taxon, taxon,conf):
        """This is de constructor of Taxon class

        Args:
            name_taxon (str): the name of the taxon, for example: 'class', 'order'...
            taxon (str): the value of the name taxon, for example: 'Gasteropoda'.
            conf (float): the value of conf that predict model returns.
        """
        self.name_taxon = name_taxon
        self.taxon = taxon
        self.conf = conf
            
class Animal:
    
    def __init__(self):
        self.taxonomic_path_top1 = {}
        self.taxonomic_path_top2 = {}
        self.taxonomic_path_top3 = {}
        self.taxonomic_path_top4 = {}
        self.taxonomic_path_top5 = {}
        self.taxonomic_paths = [
            self.taxonomic_path_top1,
            self.taxonomic_path_top2,
            self.taxonomic_path_top3,
            self.taxonomic_path_top4,
            self.taxonomic_path_top5
        ]
    
    def add_taxon(self, names_taxon_and_taxons_and_confs):
        for i, (name_taxon, taxon, conf) in enumerate(names_taxon_and_taxons_and_confs[:5]):
            self.taxonomic_paths[i][name_taxon] = Taxon(name_taxon, taxon, conf)
        
    def get_conf(self):
        conf = None
        for _,taxon in self.taxonomic_path_top1.items():
            if conf:
                conf = conf * taxon.conf
            else:
                conf = taxon.conf
        return conf, self.taxonomic_path_top1
                    
                
            

class Predict:
    animals = {}
    
    
        
    def precision_predict(self,images):
        """This function only gave one animal and use all images to try to predict with more precision

        Args:
            imagenes (list(str)): list of path to images that will be predicted
        """
        animals = self.simple_predict(images)
        
        animals_dict = {}
        last_taxon_name = TAXONOMIC_RANKS[-1]
        for animal,image in animals:
            conf, taxonomic_path = Animal(animal).get_conf()
            last_taxon = taxonomic_path[last_taxon_name]
            if last_taxon not in animals_dict:
                animals_dict[last_taxon] = conf
            else: 
                animals_dict[last_taxon] += conf
        
        tops = []
        for animal in animals:
            animal = Animal(animal)
            conf,taxon_path_top1 = animal.get_conf()
            tops.append((conf,taxon_path_top1))
            
        # Ordenar la lista tops por conf en orden descendente
        tops.sort(key=lambda x: x[0], reverse=False)
        return tops
            
    
    
    def simple_predict(self,images):
        """This function gave one animal per image giving a process to predict a lot of animals in only one funtion. Less precision. 

        Args:
            imagenes (list(str)): list of path to images that will be predicted
        """
        animals = []
        for image in images:
            animal = Animal(self.predict_image(image))
            animals.append((animal,image))
        return animals
    
    def preditc_taxon(self, image,model,name_taxon):
        results = model.predict(image,verbose=VERBOSE,device='cpu')
        
        # Assuming `results` is a list of `Results` objects
        for result in results:
            # Access bounding boxes, confidences, and classes
            probs = result.probs  # Probs object for classification outputs

            top5_confs = probs.top5conf
            top5_class_indexs = probs.top5
            names = result.names
            
            return_results = []
            for conf,index in zip(top5_confs,top5_class_indexs):
                taxon = names[index]
                return_results.append((name_taxon, taxon, conf))
            
            return return_results
        
    def predict_image(self,image):
        animal = Animal()
        
        folder = os.path.join(PATH_MODELS_TRAINED,"model_g")
        model_frist = YOLO(folder)
        names_taxon_and_taxons_and_confs = self.preditc_taxon(image,model_frist,TAXONOMIC_RANKS[0])
        animal.add_taxon(names_taxon_and_taxons_and_confs)
        taxon = names_taxon_and_taxons_and_confs[0][0] # the taxon who has the most conf
        
        for index in range(1,len(TAXONOMIC_RANKS)):
            taxon_name = TAXONOMIC_RANKS[index-1]
            folder = os.path.join(folder,f"{taxon_name}_{taxon}")
            model = YOLO(folder)
            names_taxon_and_taxons_and_confs = self.preditc_taxon(image,model,TAXONOMIC_RANKS[index])
            taxon = names_taxon_and_taxons_and_confs[0][0] # the taxon who has the most conf
            animal.add_taxon(names_taxon_and_taxons_and_confs)
                  
        animal.images.append(image)
        return animal
            

    def predecir(self,args:str, precision_predict= False):
        images = args.split(',')
        if precision_predict:
            self.precision_predict(images)
        
        
        

if __name__ == "__main__":
    if (len(sys.argv) <= 1):
        fail("Falta argumento de la imagen\npython3 yolo_predict.py 'img_path'")
        exit(-1)
    
    else:
        predict = Predict()
        predict.predecir()
        # predecir(sys.argv[1])