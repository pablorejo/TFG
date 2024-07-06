from conf import (
    VERBOSE,
    info,
    warning,
    fail,
    IMAGE_SIZE,
    types,
    MAX_NUM_OF_CROPS,
    DEVICE,
    BATCH,
    NUM_WORKERS
)
import torch
import cv2
from imgaug import augmenters as iaa
import imageio
import requests
from requests.exceptions import RequestException, Timeout
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from multiprocessing import cpu_count
import os

def improve_contrast(image):
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel back with a and b channels
    limg = cv2.merge((cl, a, b))

    # Convert image back to BGR color space
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)



def denoise_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def augment_image(image_path: str, number: int):
    # Define an augmentation sequence
    augmentation_sequence = iaa.Sequential([
        iaa.Fliplr(0.5),  # Horizontal flip with 50% probability
        iaa.Crop(percent=(0, 0.1)),  # Crop the images between 0% to 10%
        iaa.GaussianBlur(sigma=(0, 3.0)),  # Apply Gaussian blur with sigma between 0 and 3.0
        iaa.Multiply((0.8, 1.2)),  # Change brightness by multiplying values between 0.8 and 1.2
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Scale on x and y axis between 80% and 120%
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Translate on x and y axis between -20% and 20%
            rotate=(-25, 25),  # Rotate images between -25 and 25 degrees
            shear=(-8, 8)  # Apply shear between -8 and 8 degrees
        )
    ], random_order=True)  # Apply augmentations in random order

    # Load the image
    image_in = cv2.imread(image_path)
    
    if image_in is None:
        warning(f"Could not load image from path: {image_path}")
        return

    # Denoise and improve contrast
    image_denoise = denoise_image(image_in)
    image_contrast = improve_contrast(image_denoise)
    image = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Apply the augmentations
    image_aug = augmentation_sequence(image=image)

    # Resize the image to the defined size
    image_aug_resized = cv2.resize(image_aug, (IMAGE_SIZE, IMAGE_SIZE))

    # Save the augmented image
    image_path_out = image_path.replace(".webp", f"_{number}.webp")
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
    
    Returns:
    str: The path to the converted image.
    """
    if not output_image_path:
        output_image_path = input_image_path.rsplit(".", 1)[0] + ".webp"

    try:
        with Image.open(input_image_path) as image:
            img_adjusted = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            if not only_rescale:
                img_adjusted.save(output_image_path, 'WEBP', quality=quality)
                info(f"Image successfully converted: {output_image_path}")
                if remove_original:
                    os.remove(input_image_path)
            img_adjusted.close()
    except UnidentifiedImageError:
        warning(f"Unidentifiable image: {input_image_path}, proceeding to delete.")
        if os.path.exists(input_image_path):
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
    except Exception as e:
        fail(f"Unexpected error checking image {image_path}: {e}, setting corrupt to True")
        return True

def discard_bad_image(img_path, model_to_discard, ask=False, confidence=0.90):
    """
    Discards images that are not useful for the project according to the model already trained by YOLO.

    Args:
    img_path (str): The path to the image to predict.
    ask (bool): Whether to ask if confidence is less than 85%, default is False.
    confidence (float): The confidence level you want the model to have between (0, 1), default is 0.90.

    Returns:
    bool: True if the image is good, False if it is bad or not a file.
    """
    try:
        # Make a prediction on the image
        results = model_to_discard.predict(img_path, verbose=VERBOSE, device='cpu')
        
        for result in results:
            probs = result.probs  # Probs object for classification outputs
            top1_class_index = probs.top1
            top1_confidence = probs.top1conf

            if result.names[top1_class_index] == types['bad']:
                if os.path.isfile(img_path) or os.path.islink(img_path):
                    if top1_confidence < confidence and ask:
                        with Image.open(img_path) as image:
                            image.show()
                            print(f"Most probable class: {result.names[top1_class_index]} with confidence {top1_confidence}\n")
                            response = input('Do you want to delete the image? (y/n)\n').lower()
                            if response == 'y':
                                os.remove(img_path)
                                info("You indicated the image is bad")
                                return False
                            else: 
                                info("You indicated the image is good")
                                return True
                    else: 
                        os.remove(img_path)
                        warning("The image is bad")
                        return False
                else:
                    fail("The image is not a file") 
                    return False
            else: 
                info("The image is good")
                return True
    except NameError as e:
        warning(f"The discard model does not exist, so returning true: {e}")
        return True
    except Exception as e:
        warning(f"Unexpected error: {e}")
        return True

def crop_images(src_img: str, model_to_crop: YOLO, model_to_discard: YOLO, delete_original: bool = True):
    """
    Crops images based on an AI model.

    Args:
    src_img (str): Path to the image.
    model_to_crop (YOLO): Model for detection.
    model_to_discard (YOLO): Model to discard bad images.
    delete_original (bool): Whether to delete the original image after cropping.

    Returns:
    list: An array with the paths of the cropped images.
    """
    image = cv2.imread(src_img)
    if image is None:
        info(f"Error reading image: {src_img}")
        return []

    try:
        results = model_to_crop(image, verbose=VERBOSE, device='cpu')
        base_name, _ = os.path.splitext(src_img)
        number = 0
        paths = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                if bool(box.conf > 0.8):
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    cropped_img = image[y_min:y_max, x_min:x_max]
                    cropped_img_path = f'{base_name}_cropped_{number}.jpg'
                    cv2.imwrite(cropped_img_path, cropped_img)

                    if discard_bad_image(cropped_img_path, model_to_discard):
                        paths.append(cropped_img_path)
                        number += 1
                        info(f'Cropped image saved at: {cropped_img_path}')
                        if len(paths) == MAX_NUM_OF_CROPS:
                            break
                    else:
                        os.remove(cropped_img_path)
            if len(paths) == MAX_NUM_OF_CROPS:
                break

    except Exception as e:
        warning(f"Error processing image {src_img}: {str(e)}")
        return []

    finally:
        if delete_original:
            os.remove(src_img)
        else:
            paths.append(src_img)

    return paths

def download_image(url, full_path, max_retries=3, timeout=5):
    """
    Download an image and save it.

    Args:
        url (str): URL to the image to download.
        full_path (str): Path where the image will be saved.
        max_retries (int, optional): Number of times to retry the download. Defaults to 3.
        timeout (int, optional): Time to wait for the download in seconds. Defaults to 5.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                with open(full_path, 'wb') as file:
                    file.write(response.content)
                with Image.open(full_path) as img:
                    img.verify()
                info(f"Image {full_path} downloaded correctly")
                return True
        except (IOError, SyntaxError) as e:
            warning(f"Invalid image file: {full_path}")
            if os.path.exists(full_path):
                os.remove(full_path)
            break
        except (RequestException, Timeout) as e:
            warning(f"URL error: {url} (attempt {attempt+1}/{max_retries}) - {str(e)}")
        finally:
            if 'response' in locals():
                response.close()
    
    fail(f"Image with URL error: {url}, could not be downloaded")
    return False

def train_yolo_model(model, model_name, train_folder_path, model_folder, epochs):
    """
    Train a YOLO model based on configurations set in conf.py and parameters received.

    Args:
        model (YOLO): The YOLO model to train.
        model_name (str): Name to save the model.
        train_folder_path (str): Path where training images are located.
        model_folder (str): Path where training results will be saved.
        epochs (int): Number of training epochs.

    Returns:
        results: Training results.
    """
    info(f"Starting training: {model_name} {train_folder_path} {model_folder}")
    
    num_cores = cpu_count()
    device = DEVICE  # Ensure DEVICE is set properly, you may need to import or define this

    if device == 'cpu':
        os.environ['OMP_NUM_THREADS'] = str(num_cores)
        torch.set_num_threads(num_cores)
        model.to('cpu')
        info("Using CPU for training")
    else:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        model.to(device)
        info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    try:
        results = model.train(
            data=train_folder_path,
            epochs=epochs,
            imgsz=IMAGE_SIZE,  # Ensure IMAGE_SIZE is defined
            name=model_name,
            project=model_folder,
            batch=BATCH,  # Ensure BATCH is defined
            device=device,
            amp=(device != 'cpu'),  # Automatic Mixed Precision (AMP) only for GPU
            workers=NUM_WORKERS  # Ensure NUM_WORKERS is defined
        )
        return results
    except RuntimeError as e:
        info(f"Error during training: {e}")
        return None