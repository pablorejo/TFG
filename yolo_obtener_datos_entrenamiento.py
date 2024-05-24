from defs import *

def get_model_data(model_path: str):
    pass

def get_models(models_path: str):
    folders = get_folders_by_level(PATH_MODELS_TRAINED, 1)
    pass

def process_data():
    models = get_models()
    
    data = []
    for model in models:
        name, info, rank = get_model_data(model)
        information = {
            'info': info,
            'name': name,
            'rank': rank
        }
        data.append(information)

if __name__ == "__main__":
    process_data()
