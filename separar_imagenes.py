import os
import tkinter as tk
from PIL import Image, ImageTk  # Correct import
import random
from conf import *
from defs import find_images

"""This file is used to create two datasets of images for the training of good and bad images. It is need to create a model that can discard automatically bad images"""

class ImageOrganizerApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Image Organizer")

        self.clear_files()
        
        self.jpg_images = find_images(DISCARD_IMAGE_PATH, extensions=['.webp','.jpg'])
        random.shuffle(self.jpg_images)
        self.image_number = 0

        self.label = tk.Label(master)  # Create the label once
        self.label.pack()

        master.bind('<Left>', self.left_button_action)  # Corrected: remove the ()
        master.bind('<Right>', self.right_button_action)  # Corrected: remove the ()

        self.add_buttons()
        self.next_image()

    def next_image(self):
        if self.image_number >= len(self.jpg_images):
            print("No more images.")
            return

        self.url = self.jpg_images[self.image_number]
        print(self.url)
        self.image = Image.open(self.url)

        # Desired size (keep these values or adjust to your need)
        max_width = 800
        max_height = 600

        # Calculate aspect ratio and new size while maintaining aspect ratio
        original_width, original_height = self.image.size
        ratio = min(max_width / original_width, max_height / original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))

        # Resize the image
        self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(self.image) 
        
        self.label.configure(image=self.photo)  # Update the image of the existing label
        self.label.image = self.photo  # Keep a reference
        self.image_number += 1

    def add_buttons(self):
        button_frame = tk.Frame(self.master)
        button_frame.pack(side='bottom', fill='x', pady=10)

        btn_left = tk.Button(button_frame, text="Bad", command=self.left_button_action)
        btn_left.pack(side='left', padx=20)

        btn_right = tk.Button(button_frame, text="Good", command=self.right_button_action)
        btn_right.pack(side='right', padx=20)

    def left_button_action(self, event=None):
        with open(BAD_IMAGE_FILE, 'a', encoding='utf-8') as file:
            file.write(self.url + "\n")
        self.next_image()

    def right_button_action(self, event=None):
        with open(GOOD_IMAGE_FILE, 'a', encoding='utf-8') as file:
            file.write(self.url + "\n")
        self.next_image()
    
    def clear_files(self):
        with open(BAD_IMAGE_FILE, 'w', encoding='utf-8') as file:
            file.close()
        with open(GOOD_IMAGE_FILE, 'w', encoding='utf-8') as file:
            file.close()

if __name__ == "__main__":
    root = tk.Tk()
    my_gui = ImageOrganizerApp(root)
    root.mainloop()
