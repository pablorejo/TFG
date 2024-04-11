from globales import *
import os
path = 'runs\\classify\\modelo\\weights\\best.pt'
print(os.path.exists(path))
data_yaml = 'data.yaml'
model = YOLO(path)
model.val(data=data_yaml)