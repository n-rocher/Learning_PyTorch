import os
import cv2
import numpy as np

import torch
from PIL import Image
from torchvision.transforms import ToTensor, Compose

AUDI_A2D2_CATEGORIES = {
    1: {"name": "Road", "color": [[180, 50, 180], [255, 0, 255]]},
    2: {"name": "Lane", "color": [[255, 193, 37], [200, 125, 210], [128, 0, 255]]},
    3: {"name": "Crosswalk", "color": [[210, 50, 115]]},
    4: {"name": "Curb", "color": [[128, 128, 0]]},
    5: {"name": "Sidewalk", "color": [[180, 150, 200]]},

    6: {"name": "Traffic Light", "color": [[0, 128, 255], [30, 28, 158], [60, 28, 100]]},
    7: {"name": "Traffic Sign", "color": [[0, 255, 255], [30, 220, 220], [60, 157, 199]]},

    8: {"name": "Person", "color": [[204, 153, 255], [189, 73, 155], [239, 89, 191]]},

    9: {"name": "Bicycle", "color": [[182, 89, 6], [150, 50, 4], [90, 30, 1], [90, 30, 30]]},
    10: {"name": "Bus", "color": []},
    11: {"name": "Car", "color": [[255, 0, 0], [200, 0, 0], [150, 0, 0], [128, 0, 0]]},
    12: {"name": "Motorcycle", "color": [[0, 255, 0], [0, 200, 0], [0, 150, 0]]},
    13: {"name": "Truck", "color": [[255, 128, 0], [200, 128, 0], [150, 128, 0], [255, 255, 0], [255, 255, 200]]},

    14: {"name": "Sky", "color": [[135, 206, 255]]},
    15: {"name": "Nature", "color": [[147, 253, 194]]},
    16: {"name": "Building", "color": [[241, 230, 255]]}
}

CATEGORIES_COLORS = {
    1: {"name": "Road", "color": [75, 75, 75]},
    2: {"name": "Lane", "color": [255, 255, 255]},
    3: {"name": "Crosswalk", "color": [200, 128, 128]},
    4: {"name": "Curb", "color": [150, 150, 150]},
    5: {"name": "Sidewalk", "color": [244, 35, 232]},

    6: {"name": "Traffic Light", "color": [250, 170, 30]},
    7: {"name": "Traffic Sign", "color": [255, 255, 0]},

    8: {"name": "Person", "color": [255, 0, 0]},

    9: {"name": "Bicycle", "color": [88, 41, 0]},
    10: {"name": "Bus", "color": [255, 15, 147]},
    11: {"name": "Car", "color": [0, 255, 142]},
    12: {"name": "Motorcycle", "color": [0, 0, 230]},
    13: {"name": "Truck", "color": [75, 10, 170]},

    14: {"name": "Sky", "color": [135, 206, 255]},
    15: {"name": "Nature", "color": [107, 142, 35]},
    16: {"name": "Building", "color": [241, 230, 255]}
}


values = CATEGORIES_COLORS.values()
COLOR_CATEGORIES = np.zeros((len(values), 3), dtype=np.uint8)
for i, data in enumerate(values):
    COLOR_CATEGORIES[i] = data["color"]

class A2D2_Dataset(torch.utils.data.Dataset):

    def __init__(self, type, size=(418, 418)):

        assert type in ["testing", "training", "validation"], 'add the folder type of data "testing", "training", or "validation"'

        self.input_size = size

        self.dataset_folder = r"F:\\A2D2 Camera Semantic\\" + type + "\\"
        self.input_img_paths, self.target_img_paths = self.getData()

        self.CATEGORIES = AUDI_A2D2_CATEGORIES

        self.img_transform = Compose([ToTensor()])

    @staticmethod
    def classes():
        return len(CATEGORIES_COLORS)

    def name(self):
        return "A2D2Dataset"

    def getData(self):
        '''
        Find files of A2D2
        '''
        data_image = []
        data_label = []

        camera_day_folders = [os.path.join(self.dataset_folder, item) for item in os.listdir(self.dataset_folder) if os.path.isdir(self.dataset_folder + item)]
        for folder in camera_day_folders:
            camera_files_folder = os.path.join(folder, "camera", "cam_front_center")
            label_files_folder = os.path.join(folder, "label", "cam_front_center")

            camera_files_files = [os.path.join(camera_files_folder, file) for file in os.listdir(camera_files_folder)]
            label_files_files = [os.path.join(label_files_folder, file) for file in os.listdir(label_files_folder)]

            data_image = data_image + camera_files_files
            data_label = data_label + label_files_files

        return data_image, data_label

    def __len__(self):
        return len(self.target_img_paths)

    def __getitem__(self, idx):

        input_img_path = self.input_img_paths[idx]
        target_img_path = self.target_img_paths[idx]

        img = Image.open(input_img_path).convert("RGB")
        img = img.resize(self.input_size)


        # img = np.asarray(img, dtype=np.uint8)

        target_raw = Image.open(target_img_path)
        target_raw = target_raw.resize(self.input_size)
        target_raw = np.asarray(target_raw)

        # instance = np.zeros(self.input_size[::-1])
        target = np.zeros((len(AUDI_A2D2_CATEGORIES)+ 1,) + self.input_size[::-1])

        # For every categories in the list
        for i, id_category in enumerate(self.CATEGORIES):

            data_category = self.CATEGORIES[id_category]

            for color in data_category["color"]:

                # We select pixels belonging to that category
                test = cv2.inRange(target_raw, tuple(color), tuple(color))

                # Add the value where it belongs to
                target[i+1] = target[i+1] + (test >= 1)

                # We copy 255 value for a white image
                # res = cv2.bitwise_and(ins_255, ins_255, mask=test)

                # And we past it to the good id to the instance
                # instance = instance + res
                instance = instance + test

        target[0] = instance == 0

        # Transform to pytorch tensor
        img = self.img_transform(img)
        target = torch.from_numpy(target)

        return img, target