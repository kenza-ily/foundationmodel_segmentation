import os

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageFilter

image_size = None

class PetSegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, image_transform=None, mask_transform=None,
                 rotate=False, adjust_brightness=False, color_jitter=False, noise_injection=False, perspective=False, motion_blur=False):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.rotate = rotate
        self.adjust_brightness = adjust_brightness
        self.color_jitter = color_jitter
        self.noise_injection = noise_injection
        self.perspective = perspective
        self.motion_blur = motion_blur

        self.images = [img for img in os.listdir(image_dir) if img.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.annotation_dir, self.images[idx].replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")  # Assuming mask is single-channel

        mask_np = np.array(mask_pil)  # Convert PIL Image to numpy for inspection
        mask_np = np.where(mask_np == 1, 1, 0).astype(np.uint8)  # assuming you want to convert values 2 and 3 to 1
        mask_pil = Image.fromarray(mask_np)

        image, mask_pil = self.perturb_image(image, mask_pil)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask_tensor = self.mask_transform(mask_pil)

        return image, mask_tensor

    def perturb_image(self, image, mask):
        if self.rotate:
            angle = np.random.uniform(-30, 30)  # Random rotation between -30 and 30 degrees
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)

        if self.adjust_brightness:
            brightness_factor = np.random.uniform(0.5, 1.5)
            image = transforms.functional.adjust_brightness(image, brightness_factor)

        if self.color_jitter:
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            image = jitter(image)

        if self.noise_injection:
            noise = torch.randn_like(transforms.functional.to_tensor(image)) * 0.05
            image = transforms.functional.to_tensor(image) + noise
            image = transforms.functional.to_pil_image(image.clamp(0, 1))

        if self.perspective:
            perspective_transform = transforms.RandomPerspective(distortion_scale=0.5, p=1, interpolation=3)
            image = perspective_transform(image)
            mask = perspective_transform(mask)

        if self.motion_blur:
            kernel_size = int(np.random.uniform(5, 10))
            image = image.filter(ImageFilter.GaussianBlur(radius=kernel_size))

        return image, mask
    
def getPetDataset(image_transform, mask_transform, args):
    # INSERT PATH HERE
    pet_dataset = PetSegmentationDataset(image_dir='/cs/student/projects1/2020/ssoomro/24UCL_SelfSupervised_Segmentation/mae/pet_dataset/images', annotation_dir='/cs/student/projects1/2020/ssoomro/24UCL_SelfSupervised_Segmentation/mae/pet_dataset/annotations/trimaps',image_transform=image_transform, mask_transform=mask_transform)

    # Determine the lengths for train and validation sets
    total_count = len(pet_dataset)
    test_count = int(0.2 * total_count)
    val_count = int(0.1 * total_count)

    train_count = total_count - test_count - val_count  # the rest for training

    # Split the dataset into train, validation, and test sets
    train_dataset, temp_dataset = random_split(pet_dataset, [train_count, total_count - train_count], generator=torch.Generator().manual_seed(args.seed))
    val_dataset, test_dataset = random_split(temp_dataset, [val_count, test_count], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_dataset, val_dataset, test_dataset    

class OxfordPetsDataset(Dataset):
    def __init__(self, img_dir, annotation_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = self._load_annotations(annotation_file)

    def _load_annotations(self, annotation_file):
        annotations = {}
        with open(annotation_file, 'r') as file:
            for line in file.readlines():
                if 'CLASS-ID' in line:
                    continue
                parts = line.strip().split(' ')
                image_filename = parts[0] + '.jpg'
                try:
                    class_id = int(parts[1]) - 1
                    annotations[image_filename] = class_id
                except ValueError:
                    print(f"Skipping line due to format issue: {line}")
                    continue
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = list(self.annotations.keys())[idx]
        class_id = self.annotations[img_name]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_id

def set_image_size(args):
    global image_size
    image_size = args.image_size

def custom_resize(mask):
    # print("Before resizing - unique values:", np.unique(np.array(mask, dtype=np.uint8)))
    mask = transforms.Resize((image_size, image_size))(mask)
    return mask

def custom_to_tensor(mask):
    # print("Before ToTensor - unique values:", np.unique(np.array(mask, dtype=np.uint8)))
    mask = transforms.ToTensor()(mask)
    return mask

def custom_to_tensor(mask):
    # Convert mask to tensor manually without scaling by 255
    return torch.from_numpy(np.array(mask, dtype=np.uint8)).float().div(255)

def custom_threshold(mask):
    mask = torch.where(mask > 0, torch.tensor(1.0), torch.tensor(0.0))
    return mask
    
def getPetDataset(args):
    imgs_mean = [0.485, 0.456, 0.406]
    imgs_std = [0.229, 0.224, 0.225]
    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(imgs_mean, imgs_std)
    ])
    set_image_size(args)
    mask_transform = transforms.Compose([
    transforms.Lambda(custom_resize),
    transforms.Lambda(custom_to_tensor),
    transforms.Lambda(custom_threshold),
    ])
    
    # INSERT PATH HERE
    pet_dataset = PetSegmentationDataset(image_dir='/cs/student/projects1/2020/ssoomro/24UCL_SelfSupervised_Segmentation/mae/pet_dataset/images', annotation_dir='/cs/student/projects1/2020/ssoomro/24UCL_SelfSupervised_Segmentation/mae/pet_dataset/annotations/trimaps',image_transform=image_transform, mask_transform=mask_transform)

    # Determine the lengths for train and validation sets
    total_count = len(pet_dataset)
    test_count = int(0.2 * total_count)
    val_count = int(0.1 * total_count)

    train_count = total_count - test_count - val_count  # the rest for training

    # Split the dataset into train, validation, and test sets
    train_dataset, temp_dataset = random_split(pet_dataset, [train_count, total_count - train_count], generator=torch.Generator().manual_seed(args.seed))
    val_dataset, test_dataset = random_split(temp_dataset, [val_count, test_count], generator=torch.Generator().manual_seed(args.seed))

    if args.fine_tune_size != 1:
        # Calculate the number of samples to use from the train_dataset
        train_use_count = int(args.fine_tune_size * len(train_dataset))
        train_unused_count = len(train_dataset) - train_use_count
        # Split the train dataset to only use the specified portion
        train_dataset, _ = random_split(train_dataset,[train_use_count, train_unused_count],generator=torch.Generator().manual_seed(args.seed))
    
    print(f"Number of PET training samples: {len(train_dataset)}")
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # return train_dataset, val_dataset, test_dataset
    return train_loader, val_loader, test_loader