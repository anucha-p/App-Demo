import logging
import numpy as np
import torch
import pydicom as dicom
import nibabel as nib
from multiprocessing import Pool
from torch.utils.data import Dataset
from os.path import splitext, isfile, join
from os import listdir
from pathlib import Path
from tqdm import tqdm
from functools import partial
from skimage.transform import resize
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == ".npy":
        return np.load(filename)
    elif ext in [".pt", ".pth"]:
        return torch.load(filename).numpy()
    elif ext in [".gz", ".nii"]:
        return nib.load(filename).get_fdata()
    elif ext in [".dcm", ".DCM"]:
        return dicom.dcmread(filename).pixel_array
    else:
        return np.asarray(Image.open(filename))


class PolarDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        target_file: str,
        scale: float = 1.0,
        is_train: bool = False,
    ):
        self.images_dir = Path(images_dir)
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.is_train = is_train
        dataset = pd.read_csv(target_file)
        # self.label_df = dataset.loc[:, ["id", "LAD", "LCX", "RCA"]].copy()
        self.label_df = dataset.loc[:, ["id", "Positive", "Negative"]].copy()
        # self.transform = transform

        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith(".")
        ]
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )

        logging.info(f"Creating dataset with {len(self.ids)} examples")
        # logging.info('Scanning mask files to determine unique values')
        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #         total=len(self.ids)
        #     ))

        # self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def transform(image, scale):
        w, h, *_ = image.shape
        # print(image.shape)
        newW, newH = int(scale * w), int(scale * h)
        assert (
            newW > 0 and newH > 0
        ), "Scale is too small, resized images would have no pixel"

        # # color/intensity augmentation
        # if image.ndim == 2:
        #     image = np.swapaxes(image, -2, -1)

        # if image.ndim == 3:
        #     image = np.swapaxes(image, 2, 0)
        # image = image[np.newaxis, ...]
        img2 = np.zeros((3, w, h))
        img2[0, :, :] = image
        img2[1, :, :] = image
        img2[2, :, :] = image
        image = torch.tensor(img2)
        image = (image - image.min()) / (image.max() - image.min())

        # t_resize = transforms.Resize(size=(224,224), antialias=True)
        # image = t_resize(image)

        # if random.random() > 0.5:
        #     tf_resize = transforms.Resize(size=(70,70))
        #     image = tf_resize(image)
        #     mask = tf_resize(mask)

        #     # Random crop
        #     i, j, h, w = transforms.RandomCrop.get_params(
        #         image, output_size=(64, 64))
        #     image = TF.crop(image, i, j, h, w)
        #     mask = TF.crop(mask, i, j, h, w)

        # if random.random() > 0.5:
        #     angle = random.randint(-15, 15)
        #     image = TF.rotate(image, angle)
        #     mask = TF.rotate(mask, angle)

        return image

    def __getitem__(self, idx):
        name = self.ids[idx]
        # mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + ".*"))

        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {name}: {img_file}"
        img = load_image(img_file[0])
        label = self.label_df.loc[
            self.label_df["id"] == name, ["Positive", "Negative"]
        ].to_numpy()
        label = torch.tensor(label[0])
        # print(label)

        # assert img.size == mask.size, \
        #     f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.transform(image=img, scale=self.scale)

        # img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        # mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        sample = {
            "image": img.float().contiguous(),
            "label": label.float().contiguous(),
        }
        # if self.transform:
        #     sample = self.transform(sample)
        # else:
        #     sample = {
        #     'image': torch.as_tensor(img.copy()).float().contiguous(),
        #     'mask': torch.as_tensor(mask.copy()).long().contiguous()
        # }

        return sample
