"""SegmentationData Dataloader"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter

from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import torch.utils.data as data

class SegmentationData(Dataset):
    """Custom Segmentation Dataset.
    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/citys'
    split: string
        'train', 'valid' or 'test'
    transform : callable, optional
        A function that transforms the image
    """
    # BASE_DIR = 'segdata'
    NUM_CLASS = 3

    def __init__(self, images_dir, masks_dir, nb_classes, mode=None,
        transform=None, base_size=480, crop_size=480, **kwargs):
    
        super(SegmentationData, self).__init__()
        self.mode = mode
        self.nb_classes=nb_classes
        self.ids = os.listdir(images_dir)
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.images = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks = [os.path.join(masks_dir, image_id.split('.')[0] + '.npy') \
            for image_id in self.ids]
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + images_dir + "\n")
        # self.valid_classes = [0, 1, 2]

    # def _class_to_index(self, mask):
    #     values = np.unique(mask)
    #     for value in values:
    #         assert (value in self._mapping)
    #     index = np.digitize(mask.ravel(), self._mapping, right=True)
    #     return self._key[index].reshape(mask.shape)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = np.load(self.masks[index])
        mask[mask > self.nb_classes - 1] = 0 # set other class to zero
        mask = Image.fromarray(mask)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'valid':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # print(type(img)SegmentationData)
        # final transformSegmentationData
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        # target = self._class_to_index(np.array(mask).astype('int32'))
        # return torch.LongTensor(np.array(target).astype('int32'))
        return torch.LongTensor(np.array(mask).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        """Number of categories."""
        return self.nb_classes


# def _get_city_pairs(folder, split='train'):
#     def get_path_pairs(img_folder, mask_folder):
#         img_paths = []
#         mask_paths = []
#         for root, _, files in os.walk(img_folder):
#             for filename in files:
#                 if filename.endswith(".png"):
#                     imgpath = os.path.join(root, filename)
#                     foldername = os.path.basename(os.path.dirname(imgpath))
#                     maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
#                     maskpath = os.path.join(mask_folder, foldername, maskname)
#                     if os.path.isfile(imgpath) and os.path.isfile(maskpath):
#                         img_paths.append(imgpath)
#                         mask_paths.append(maskpath)
#                     else:
#                         print('cannot find the mask or image:', imgpath, maskpath)
#         print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
#         return img_paths, mask_paths

#     if split in ('train', 'val'):
#         img_folder = os.path.join(folder, 'leftImg8bit/' + split)
#         mask_folder = os.path.join(folder, 'gtFine/' + split)
#         img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
#         return img_paths, mask_paths
#     else:
#         assert split == 'trainval'
#         print('trainval set')
#         train_img_folder = os.path.join(folder, 'leftImg8bit/train')
#         train_mask_folder = os.path.join(folder, 'gtFine/train')
#         val_img_folder = os.path.join(folder, 'leftImg8bit/val')
#         val_mask_folder = os.path.join(folder, 'gtFine/val')
#         train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
#         val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
#         img_paths = train_img_paths + val_img_paths
#         mask_paths = train_mask_paths + val_mask_paths
#     return img_paths, mask_paths

def mask_to_image(mask):
    h = mask.shape[0]
    w = mask.shape[1]
    mask_rgb = Image.new('RGB', (w, h))
    for j in range(0, h):
        for i in range(0, w):
            pixal = mask[j, i]
            if pixal == 0:
                mask_rgb.putpixel((i, j), (61,10,81))
            elif pixal == 1:
                mask_rgb.putpixel((i, j), (69,142,139))
            elif pixal == 2:
                mask_rgb.putpixel((i, j), (250,231,85))
            else:
                mask_rgb.putpixel((i, j), (255, 255, 255))
    return mask_rgb

if __name__ == '__main__':

    nb_classes = 3

    train_img_dir = '/home/wangjialei/teeth_dataset/new_data_20190621/train_new/images'
    train_mask_dir = '/home/wangjialei/teeth_dataset/new_data_20190621/train_new/masks'

    valid_img_dir = '/home/wangjialei/teeth_dataset/new_data_20190621/valid_new/images'
    valid_mask_dir = '/home/wangjialei/teeth_dataset/new_data_20190621/valid_new//masks'

    train_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.519401, 0.359217, 0.310136], [0.061113, 0.048637, 0.041166]),#R_var is 0.061113, G_var is 0.048637, B_var is 0.041166
    ])
    valid_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.517446, 0.360147, 0.310427], [0.061526,0.049087, 0.041330])#R_var is 0.061526, G_var is 0.049087, B_var is 0.041330
    ])

    train_set = SegmentationData(images_dir=train_img_dir, masks_dir=train_mask_dir, nb_classes=nb_classes, mode='train', transform=train_transform)
    train_loader = data.DataLoader(dataset=train_set, batch_size = 1, shuffle = True, num_workers=4, pin_memory=True)

    valid_set = SegmentationData(images_dir=valid_img_dir, masks_dir=valid_mask_dir, nb_classes=nb_classes, mode='valid', transform=valid_transform)
    valid_loader = data.DataLoader(dataset=valid_set, batch_size = 1, shuffle = False, num_workers=4, pin_memory=True)

    # train
    for iteration, (images, targets) in enumerate(train_loader):
        if iteration % 10 == 0:
            print("### proc ", iteration, " / ", len(train_loader))
        img = images[0].numpy()*255
        img = img.astype('uint8')
        img = np.transpose(img,(1,2,0))
  
        mask = targets[0]
        mask[mask > nb_classes - 1] = 0
        mask = mask_to_image(mask)

        plt.subplot(1, 2, 1)
        plt.title('image')
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.title('mask')
        plt.imshow(mask)

        save_file = "train/"
        os.makedirs(save_file, exist_ok=True)
        plt.savefig(os.path.join(save_file, str(iteration) + '.png'))
    
    #valid
    for iteration, (images, targets) in enumerate(valid_loader):
        if iteration % 10 == 0:
            print("### proc ", iteration, " / ", len(valid_loader))
        img = images[0].numpy()*255
        img = img.astype('uint8')
        img = np.transpose(img,(1,2,0))
  
        mask = targets[0]
        mask[mask > nb_classes - 1] = 0
        mask = mask_to_image(mask)

        plt.subplot(1, 2, 1)
        plt.title('image')
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.title('mask')
        plt.imshow(mask)

        save_file = "valid/"
        os.makedirs(save_file, exist_ok=True)
        plt.savefig(os.path.join(save_file, str(iteration) + '.png'))
