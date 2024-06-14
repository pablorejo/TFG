import os
import pickle
STATE_FILE = 'yolo_train_img_data.pkl'

def save_state(state):
    with open(STATE_FILE, 'wb') as f:
        pickle.dump(state, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'rb') as f:
            return pickle.load(f)
    return None

class ContextTaxon():
    def __init__(self,
                    model_folder,
                    column_filters,
                    filters,
                    taxon_index,
                    temp_image_path,
                    train_folder_path,
                    initial_counts,
                    total_counts,
                    counts_with_crops,
                    counts_with_transformations_and_crops) :
        self.model_folder = model_folder
        self.column_filters = column_filters
        self.filters = filters
        self.taxon_index = taxon_index
        self.temp_image_path = temp_image_path
        self.train_folder_path = train_folder_path
        self.initial_counts = initial_counts
        self.total_counts = total_counts
        self.counts_with_crops = counts_with_crops
        self.counts_with_transformations_and_crops = counts_with_transformations_and_crops
        
    
class SaveContext():
    
    def __init__(self):
        self.context_taxon_dict = {}
        self.end_taxon = None
    
    def add_context_taxon(self,context_taxon: ContextTaxon):
        self.context_taxon_dict[context_taxon.taxon_index] = context_taxon
        filtered_dict = {key: value for key, value in self.context_taxon_dict.items() if key <= context_taxon.taxon_index}
        self.context_taxon_dict = filtered_dict