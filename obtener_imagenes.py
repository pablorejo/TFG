import pandas as pd
from tqdm import tqdm
import math, os
import requests
from requests.exceptions import RequestException
from conf import *
from defs import *
from defs_img import *

"""
    This file is responsible for downloading all images from a dataset into the same folder.
    It automatically discards images that are not useful using an AI model.
    Additionally, it rescales and converts them to .webp format to save space.
"""

def main():
    
    # Check if running as superuser
    if os.geteuid() != 0:
        print("The program needs to be run as superuser\nsudo su")
        exit(-1)
        
    # Define the base path to save the downloaded images
    IMAGE_PATH = 'images'
    # Configure Pandas to display the full width of text columns
    pd.set_option('display.max_colwidth', None)
    # Path to the CSV file containing the occurrences
    occurrences_file_path = 'parsed_occurrences.csv'

    # Chunk size when reading the CSV file: 1 million rows at a time
    chunksize = 10 ** 4

    print("Reading the occurrences file")
    # Read the CSV file in chunks to handle large data volumes
    df_occurrences = pd.read_csv(occurrences_file_path, delimiter=',', low_memory=False, on_bad_lines='skip')

    # Calculate the total size of the file to estimate the number of chunks
    file_size = os.path.getsize(occurrences_file_path)
    # Calculate the total number of chunks based on file size and chunk size
    total_chunks = math.ceil((file_size / 166) / chunksize)

    print("Starting chunks\n")

    # Process only the first chunk to initialize the DataFrame of failed records
    df_failed = pd.DataFrame(df_occurrences.columns)
    df_completed = pd.DataFrame(df_occurrences.columns)

    print(df_occurrences.head())

    try:
        # Initialize the progress bar for total element processing
        with tqdm(total=total_chunks, desc="Processing elements", unit="element") as pbar_total:

            # Iterate over each row of the current DataFrame
            for index, row in tqdm(df_occurrences.iterrows(), desc="Processing elements", unit="elements"):

                # Check if the row has a valid identifier
                if pd.notna(row['identifier']):

                    # Build the folder path based on taxonomic classification
                    folder_path = f"{IMAGE_PATH}/{parse_name(row['class'])}/{parse_name(row['order'])}/{parse_name(row['family'])}/{parse_name(row['genus'])}/{parse_name(row['acceptedScientificName'])}"

                    # Create the folder if it doesn't exist
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path,exist_ok=True)

                    # Build the full path of the image file to save
                    full_path = os.path.join(folder_path, parse_name(str(row['gbifID'])) + ".jpg")
                    webp_path = full_path.replace(".jpg", ".webp")

                    # Download and save the image if it doesn't exist yet
                    if (not os.path.exists(full_path) or is_corrupt_image(full_path) and not os.path.exists(webp_path)):
                        try:
                            os.remove(full_path)
                            response = requests.get(row['identifier'], timeout=5)
                            if response.status_code == 200:
                                with open(full_path, 'wb', encoding='utf-8') as file:
                                    file.write(response.content)
                                    convert_to_webp(full_path)
                                    file.close()
                                # Save successfully completed records
                                df_completed = pd.concat([df_completed, pd.DataFrame([row])], ignore_index=True)
                        
                        except KeyboardInterrupt:
                            print("The program successfully terminated with errors")
                            exit(-1)

                        except RequestException as e:
                            print("URL error: " + row['identifier'])
                            print(e)
                            # Add the row to the DataFrame of failed records if an error occurs
                            df_failed = pd.concat([df_failed, pd.DataFrame([row])], ignore_index=True)
                            
                        except Exception as e: 
                            # Add the row to the DataFrame of failed records if an error occurs
                            df_failed = pd.concat([df_failed, pd.DataFrame([row])], ignore_index=True)

            # Update the progress bar after processing each chunk
            pbar_total.update(1)

        print("The program successfully completed")
        # shutdown_system()

    except Exception as e:
        print("The program successfully terminated with errors")
        print(f"Exception {e}")
        # shutdown_system()

if __name__ == '__main__':
    main()