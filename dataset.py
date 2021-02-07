import os
import random
from PIL import Image
from torch.utils.data import Dataset

class ChestXRayDataset(Dataset):
    def __init__(self, image_dirs, transform, class_names):
        '''
        Parameters
            :param image_dirs: <dict object>, key = class_names, data = directory path for each class.
            :param transform: transform to apply to images
            :param class_names: <list object> class names as string
        '''
        self.image_dirs = image_dirs
        self.transform = transform
        self.class_names = class_names

        # images = {'covid': [], 'noraml': [], 'viral': []}
        self.images = {}
        
        for class_name in class_names:
            image_list = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('.png')]
            print(f'Found {len(image_list)}images for class \'{class_name}\'')
            self.images[class_name] = image_list
    
    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])

    def __getitem__(self, index):
        # Randomly select a class
        class_name = random.choice(self.class_names)

        # Make index within range
        index = index % len(self.images[class_name])

        # Get the path of the image and open file
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')

        return self.transform(image), self.class_names.index(class_name)
