import os
from PIL import Image
from torch.utils.data import Dataset

class ExpressionDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):  # Loop through the images inside the class folder
                img_path = os.path.join(class_dir, img_name)  # Full path to the image
                self.image_paths.append(img_path)  # Add the image path to the list
                self.labels.append(label)

    def __len__(self):
        """Return the total number of samples"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
