from conf import *
import cv2
from imgaug import augmenters as iaa
import imageio
import time
import requests
from requests.exceptions import RequestException, Timeout
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from multiprocessing import cpu_count
from pathlib import Path
from torch.utils.data import Dataset

def improve_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def denoise_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def augment_image(image_path: str, number: int):

    from conf import IMAGE_SIZE, warning
    # Define an augmentation sequence
    augmentation_sequence = iaa.Sequential([
        iaa.Fliplr(0.5),  # Horizontal flip with 50% probability
        iaa.Crop(percent=(0, 0.1)),  # Crop the images between 0% to 10%
        iaa.GaussianBlur(sigma=(0, 3.0)),  # Apply Gaussian blur with sigma between 0 and 3.0
        iaa.Multiply((0.8, 1.2)),  # Change the brightness of images by multiplying values between 0.8 and 1.2
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Scale images on the x and y axis between 80% and 120%
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Translate images on the x and y axis between -20% and 20%
            rotate=(-25, 25),  # Rotate images between -25 and 25 degrees
            shear=(-8, 8)  # Apply shear to images between -8 and 8 degrees
        )
    ], random_order=True)  # Apply the augmentations in random order

    # Load the image
    image_in = cv2.imread(image_path)
    
    if image_in is None:
        warning(f"Could not load image from path: {image_path}")
        return
    
    image_denoise = denoise_image(image_in)
    image_contrast = improve_contrast(image_denoise)
    image = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for Matplotlib

    # Apply the data augmentations
    image_aug = augmentation_sequence(image=image)

    # Resize the image to 640x640
    image_aug_resized = cv2.resize(image_aug, (IMAGE_SIZE, IMAGE_SIZE))

    image_path_out = image_path.replace(".webp", f"_{number}.webp")
    # Save the augmented image
    imageio.imwrite(image_path_out, image_aug_resized)
    return image_path_out
 
def convert_to_webp(input_image_path: str, output_image_path="", quality=100, remove_original=True, only_rescale=False):
    """
    Convert an image to WebP format and resize it to a specific size.

    Args:
    input_image_path (str): The path to the input image.
    output_image_path (str): The path where the converted image will be saved.
    quality (int): The quality of the resulting image, from 1 to 100.
    remove_original (bool): Whether to delete the original image after conversion.
    only_rescale (bool): Whether to only resize the image without saving it in WebP format.
    """
    from conf import warning, info, fail, IMAGE_SIZE
    if output_image_path == "":
        output_image_path = input_image_path.split(".")[0] + ".webp"
    try:
        with Image.open(input_image_path) as image:
            img_adjusted = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            if not only_rescale:
                img_adjusted.save(output_image_path, 'WEBP', quality=quality)
                info(f"Image successfully converted: {output_image_path}")
                if remove_original:
                    image.close()
                    os.remove(input_image_path)
            else:
                image.close()

    except UnidentifiedImageError:
        warning(f"Unidentifiable image: {input_image_path}, proceeding to delete.")
        os.remove(input_image_path)

    except FileNotFoundError:
        warning(f"File not found: {input_image_path}")

    except Exception as e:
        warning(f"An exception occurred with the image: {input_image_path}\nError: {e}")
    return output_image_path

def is_corrupt_image(image_path):
    """
    Check if an image is corrupt by attempting to open it with Pillow.

    Args:
    image_path (str): The path to the image file to be checked.

    Returns:
    bool: True if the image is corrupt or cannot be identified; False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Try to verify the image
        return False  # The image opened and verified correctly
    except UnidentifiedImageError:
        return True  # The image is corrupt or not an identifiable format
    except Exception:
        return False

def discard_bad_image(img_path,model_to_discard, ask=False, confidence=0.90):
    """Discards images that are not useful for the project according to the model already trained by YOLO
    
    Args:
    img_path: The path to the image to predict
    ask: Whether to ask if confidence is less than 85%, default is False
    confidence: The confidence level you want the model to have between (0,1), default is 0.85
    
    Returns: True if the image is good, False if it is bad or not a file"""
    from conf import types, info, warning, fail, VERBOSE
    try:
        # Make a prediction on the image
        results = model_to_discard.predict(img_path,verbose=VERBOSE,device='cpu')
        
        # Assuming `results` is a list of `Results` objects
        for result in results:
            # Access bounding boxes, confidences, and classes
            probs = result.probs  # Probs object for classification outputs

            # Get the index of the top 1 class (the most probable)
            top1_class_index = probs.top1

            # Get the confidence of the top 1 class
            top1_confidence = probs.top1conf

            if result.names[top1_class_index] == types['bad']:
                if os.path.isfile(img_path) or os.path.islink(img_path):
                    
                    if top1_confidence < confidence and ask:
                        with Image.open(img_path) as image:
                            # Show the image
                            image.show()
                            print(f"Most probable class: {result.names[top1_class_index]} with confidence {top1_confidence}\n")
                            response = input('Do you want to delete the image? (y/n)\n')
                            if response == 'y':
                                os.remove(img_path)
                                info("You indicated the image is bad")
                                return False  # The image is bad
                            else: 
                                info("You indicated the image is good")
                                return True  # The image is good
                    else: 
                        os.remove(img_path)
                        warning("The image is bad")
                        return False  # The image is bad
                else:
                    fail("The image is not a file") 
                    return False  # The image is not a file
            else: 
                info("The image is good")
                return True  # The image is good
    except NameError as e:
        warning(f"The discard model does not exist, so return true: {e}")
        return True  # The discard model does not exist, so return true
    except Exception as e:
        warning(f"{e}")
        return True

def crop_images(src_img: str, model_to_crop: YOLO , model_to_discard: YOLO, delete_original: bool = True):
    """
    This function crops images based on an AI model.
    
    Args:
    src_img: Path to the image.
    model: Model for detection.
    
    Returns: An array with the paths of the cropped images.
    """
    from conf import DETECT_MODEL_PATH,chek_model, info, warning, fail,VERBOSE
    image = cv2.imread(src_img)
    if image is None:
        info(f"Error reading image: {src_img}")
        return []

    results = model_to_crop(image,verbose=VERBOSE,device='cpu')

    base_name, _ = os.path.splitext(src_img)
    number = 0
    paths = []

    for result in results:
        # Get all bounding boxes
        boxes = result.boxes

        for box in boxes:
            # Extract the bounding box coordinates, converted to integers
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            
            # Crop the image using the coordinates
            cropped_img = image[y_min:y_max, x_min:x_max]

            # Save the cropped image
            cropped_img_path = f'{base_name}_cropped_{number}.jpg'
            cv2.imwrite(cropped_img_path, cropped_img)
            
            if discard_bad_image(cropped_img_path,model_to_discard):
                number += 1
                paths.append(cropped_img_path)
                info(f'Cropped image saved at: {cropped_img_path}')
                
    if delete_original:
        os.remove(src_img)
    else:
        paths.append(src_img)
    return paths

def download_image(url, full_path, max_retries=3, timeout=5):
    """download a image and save it
    

    Args:
        url (str): url to img to download
        full_path (str): path where you want to save the image
        max_retries (int, optional): times that def will try to download. Defaults to 3.
        timeout (int, optional): tiem that download will wait for download . Defaults to 5.

    Returns:
        bool: true if download correct false if not
    """
    from conf import warning, info, fail
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                with open(full_path, 'wb') as file:
                    file.write(response.content)
                    file.close()
                    info(f"Image {full_path} correctly download")
                return True
        except (RequestException, Timeout) as e:
            warning(f"URL error: {url} (attempt {attempt+1}/{max_retries}) - {str(e)}")
            time.sleep(1)  # Exponential backoff
    fail(f'Image with URL error: {url}, could not be download')
    return False

def train_yolo_model(model: YOLO, model_name, train_folder_path, model_folder):
    info(f"{model_name} {train_folder_path} {model_folder}")
    """this def train a model based on configurations sets in conf.py and parameters recevied

    Args:
        model_name (str): name where you want to sabe model
        train_folder_path (str): path where you have training images to train the model
        model_folder (_type_): path where you want to save de results of train
    """
    num_cores = cpu_count()

    if DEVICE == 'cpu':
        os.environ['OMP_NUM_THREADS'] = str(num_cores)
        torch.set_num_threads(num_cores)
    else:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        model = model.to('cuda')  # Asegúrate de mover el modelo a la GPU

    num_workers = 8  # Puedes ajustar este número según tus necesidades

    
    try:
        results = model.train(
            data=train_folder_path,
            epochs=TRAIN_EPOCHS,
            imgsz=IMAGE_SIZE,
            name=model_name,
            project=model_folder,
            device=DEVICE,
            amp=True,
            batch=8,
            workers=num_workers
        )
        return results
    except RuntimeError as e:
        print(f"Error during training: {e}")