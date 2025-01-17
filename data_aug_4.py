import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms

# Funções de Aumento de Dados
def cutmix(data, targets, alpha=1.0, is_resnet=False, cutmix_prob=0.5):
    if np.random.rand() > cutmix_prob:
        return data, targets, targets, 1.0

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    if is_resnet:
        alpha = np.random.uniform(0.2, 0.5)

    lam = np.random.beta(alpha, alpha)

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets_a, targets_b = targets, shuffled_targets

    return new_data, targets_a, targets_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutout(data, targets, n_holes=1, length=128):
    new_data = data.clone()
    N, C, H, W = new_data.shape

    for _ in range(n_holes):
        length = np.random.randint(10, length)
        y = np.random.randint(H)
        x = np.random.randint(W)

        y1 = np.clip(y - length // 2, 0, H)
        y2 = np.clip(y + length // 2, 0, H)
        x1 = np.clip(x - length // 2, 0, W)
        x2 = np.clip(x + length // 2, 0, W)

        new_data[:, :, y1:y2, x1:x2] = 0.0

    return new_data, targets

def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets

def mixup_criterion(preds, targets):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    criterion = nn.CrossEntropyLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)

def get_da(da, DS_MEAN=[0.485, 0.456, 0.406], DS_STD=[0.229, 0.224, 0.225]):
    if da == 0: #SEM 
        data_transforms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(DS_MEAN, DS_STD)
        ])

    elif da == 1:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(DS_MEAN, DS_STD)
        ])

    elif da == 2: #COM 
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(224, 224)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(DS_MEAN, DS_STD)
        ])
        
    elif da == 3:
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])
    
    elif da == 3.1:
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomResizedCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif da == 4:
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])
    
    elif da == 4.1:
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    elif da == 4.2:
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            ###transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    elif da == 5:
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])
        
    elif da == 6:
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])

    return data_transforms
