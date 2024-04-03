import globales

# Define la ruta de la imagen a clasificar
image_paths = globales.encontrar_imagenes_jpg(globales.carpeta_de_imagenes) # Cambia esto por la ruta de tu imagen

for image_path in image_paths:
    globales.descartar_imagen_mala(img_path=image_path,preguntar=True,confianza=0.75)
                    