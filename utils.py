# utils.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageFolderCSV(Dataset):
    """
    Dataset expecting:
      - root: directory where image files live
      - csv_lines: list of (filename, label) pairs
    Label must be integer encoded (0..C-1).
    """
    def __init__(self, root, csv_lines, transform=None):
        self.root = root
        self.items = csv_lines
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, label = self.items[idx]
        path = os.path.join(self.root, fname)
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(label)

def default_transforms(image_size=224):
    train_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_t, val_t
