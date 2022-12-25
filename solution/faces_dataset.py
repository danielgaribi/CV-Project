"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

REAL_LABEL = 0
FAKE_LABEL = 1


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        if index < len(self.real_image_names):
            label = REAL_LABEL
            path = os.path.join(self.root_path, 'real', self.real_image_names[index])
        else:
            label = FAKE_LABEL
            path = os.path.join(self.root_path, 'fake', self.fake_image_names[index - len(self.real_image_names)])

        img = Image.open(path)
        if self.transform != None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        """Return the number of images in the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        return len(self.real_image_names) + len(self.fake_image_names)
