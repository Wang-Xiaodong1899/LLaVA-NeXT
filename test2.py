import datasets
import os

image_folder = "/data/RLAIF-V"

files = os.listdir(image_folder)

files = [os.path.join(image_folder, file) for file in files]

list_data_dict = datasets.load_dataset("parquet", data_files = files)["train"].cast_column("image", datasets.Image(decode=False))

item = list_data_dict[0]

import pdb; pdb.set_trace()

print(item)