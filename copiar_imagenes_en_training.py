
import globales
            
if __name__ == "__main__":
    for carpeta in globales.ruta.values():
        globales.vaciar_carpeta(carpeta)
    globales.copiar_a_training(globales.tipos['buena'],f'imagenes_{globales.tipos['buena']}.txt')
    globales.copiar_a_training(globales.tipos['mala'],f'imagenes_{globales.tipos['mala']}.txt')
    
        
    
    
        