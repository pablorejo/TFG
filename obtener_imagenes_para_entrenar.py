from conf import *
from ultralytics import YOLO
from tqdm import tqdm
import os
import pandas as pd
from requests.exceptions import RequestException
from random import randint
from functools import reduce
from defs import *
from defs_img import *
import ast


def download_random_images(n_images_to_download=1000, download_all=False):  
    """
    This function downloads a predetermined number of images from a pandas file randomly.
    
    Args:
    n_images_to_download: The number of images to be downloaded.
    download_all: Boolean indicating whether all images should be downloaded or not, 
    i.e., whether to discard bad images or not.
    """ 
    
    chunksize = 10**4
    df_occurrences = pd.read_csv(PROCESSED_DATA_CSV, chunksize=chunksize, delimiter=',', low_memory=False, on_bad_lines='skip')
    
    if SHUFFLE_DATAFRAME:
        df_occurrences = shuffle_DataFrame(df_occurrences)
    n_images = 0
    
    empty_folder(DISCARD_IMAGE_PATH)
    model_detect = chek_model(DETECT_MODEL_PATH)
    
    for chunk in df_occurrences:
        if n_images == n_images_to_download:
            break
        else:
            for row_index, row in tqdm(pd.DataFrame(chunk).iterrows(), desc="Processing elements", unit="elements"):
                if n_images < n_images_to_download:
                    # Check if the row has a valid identifier
                    if pd.notna(row['identifier']):
                        # Build the folder path based on taxonomic classification
                        folder_path = DISCARD_IMAGE_PATH

                        # Create the folder if it doesn't exist
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)

                        # Build the full path of the image file to save
                        full_path = os.path.join(folder_path, parse_name(str(row['gbifID'])) + ".jpg")
                        webp_path = full_path.replace(".jpg", ".webp")
                        
                        # Download and save the image if it doesn't exist yet
                        if (not os.path.exists(full_path) or is_corrupt_image(full_path)) and not os.path.exists(webp_path):
                            try:
                                url_list = ast.literal_eval(row['identifier'])
                                for url in url_list:
                                    if download_image(url,full_path=full_path):   
                                        if not download_all:
                                            if discard_bad_image(full_path):
                                                if model_detect is not None:
                                                    crop_images(full_path, delete_original=False)
                                                previous_path = convert_to_webp(full_path, only_rescale=True)
                                                n_images += 1
                                        else:
                                            previous_path = convert_to_webp(full_path, only_rescale=True)
                                            n_images += 1
                                            
                            except KeyboardInterrupt:
                                # save_files()
                                print("The program successfully terminated with errors")
                                exit(-1)

                            except RequestException as e:
                                print("URL error: " + row['identifier'])
                                print(e)
                                # Add the row to the DataFrame of failed records if an error occurs
                                # df_failed = pd.concat([df_failed, pd.DataFrame([row])], ignore_index=True)
                                
                            except Exception as e: 
                                pass
                                # Add the row to the DataFrame of failed records if an error occurs
                                # df_failed = pd.concat([df_failed, pd.DataFrame([row])], ignore_index=True)
                else:
                    break
                        
if __name__ == "__main__":
    download_random_images(2000, True)
