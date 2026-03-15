import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3), # Doda odporność na oświetlenie
    A.GaussNoise(p=0.2),              # Doda odporność na "cyfrowy szum"
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class ToothbrushDataset(Dataset):
    def __init__(self, root_dir, mask_dir, transform=None):
        """
        root_dir: ścieżka do 'data/train'
        mask_dir: ścieżka do 'data/ground_truth'
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Tworzymy listę wszystkich ścieżek do obrazów, przeszukując podfoldery
        self.images = []
        for class_folder in ['good', 'defective']:
            folder_path = os.path.join(root_dir, class_folder)
            for f in os.listdir(folder_path):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(class_folder, f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        rel_path = self.images[idx] # np. "defective/005.jpg"
        img_path = os.path.join(self.root_dir, rel_path)
        
        # Wyciągamy samą nazwę pliku, żeby znaleźć odpowiednią maskę
        file_name = os.path.basename(rel_path) # np. "005.jpg"
        base_name = os.path.splitext(file_name)[0]
        
        # Maska zawsze szukana jest w folderze 'defective' w ground_truth
        mask_path = os.path.join(self.mask_dir, 'defective', f"{base_name}_mask.png")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Wczytywanie maski
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.float32)
        else:
            # Jeśli to plik z folderu 'good', maska będzie pusta
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.unsqueeze(0)

#Define Transformations
train_transform = A.Compose([

    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])