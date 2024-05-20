import os
import tkinter as tk
from PIL import Image, ImageTk  # Importación correcta
import random
from globales import *
from funciones import encontrar_imagenes
"""Este fichero se usa para crear los dos datasets de las imagenes para el entrenamiento de buenas y malas
"""
class ImageOrganizerApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Image Organizer")

        self.vaciar_ficheros()
        
        self.imagenes_jpg = encontrar_imagenes("imagenes",extensions=['.webp'])
        random.shuffle(self.imagenes_jpg)
        self.numero_imagen = 0

        self.label = tk.Label(master)  # Crea el label una vez
        self.label.pack()

        master.bind('<Left>', self.left_button_action)  # Corregido: quita los ()
        master.bind('<Right>', self.right_button_action)  # Corregido: quita los ()

        self.add_buttons()
        self.siguiente_imagen()

    def siguiente_imagen(self):
        if self.numero_imagen >= len(self.imagenes_jpg):
            print("No hay más imágenes.")
            return

        self.url = self.imagenes_jpg[self.numero_imagen]
        print(self.url)
        self.image = Image.open(self.url)


         # Tamaño deseado (mantén estos valores o ajústalos a tu necesidad)
        max_width = 800
        max_height = 600

        # Calcula la relación de aspecto y el nuevo tamaño manteniendo la relación de aspecto
        original_width, original_height = self.image.size
        ratio = min(max_width/original_width, max_height/original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))

        # Redimensiona la imagen
        self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(self.image) 
        
        self.label.configure(image=self.photo)  # Actualiza la imagen del label existente
        self.label.image = self.photo  # Mantiene una referencia
        self.numero_imagen += 1

    def add_buttons(self):
        button_frame = tk.Frame(self.master)
        button_frame.pack(side='bottom', fill='x', pady=10)

        btn_left = tk.Button(button_frame, text="Mala", command=self.left_button_action)
        btn_left.pack(side='left', padx=20)

        btn_right = tk.Button(button_frame, text="Buena", command=self.right_button_action)
        btn_right.pack(side='right', padx=20)

    def left_button_action(self, event=None):
        with open(NOMBRE_ARCHIVO_MALAS, 'a') as archivo:
            archivo.write(self.url + "\n")
        self.siguiente_imagen()

    def right_button_action(self, event=None):
        with open(NOMBRE_ARCHIVO_BUENAS, 'a') as archivo:
            archivo.write(self.url + "\n")
        self.siguiente_imagen()
    
    def vaciar_ficheros(self):
        with open(NOMBRE_ARCHIVO_MALAS, 'w') as archivo:
            archivo.close()
        with open(NOMBRE_ARCHIVO_BUENAS, 'w') as archivo:
            archivo.close()
        

if __name__ == "__main__":
    root = tk.Tk()
    my_gui = ImageOrganizerApp(root)
    root.mainloop()
