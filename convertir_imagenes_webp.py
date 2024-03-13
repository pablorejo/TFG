from tqdm import tqdm  # Importa tqdm
import globales, os
imagenes = globales.encontrar_imagenes_jpg('imagenes')

print(f"Hay un total de {len(imagenes)}")
for imagen in tqdm(imagenes, desc='Convirtiendo imágenes', total=len(imagenes), ncols=100):

    globales.convert_to_webp(imagen,str(imagen).replace('.jpg','.webp'))
