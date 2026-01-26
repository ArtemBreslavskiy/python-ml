import os
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.utils.data as data
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class SegmentDataset(data.Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform

        path = os.path.join(self.path, 'images')
        list_files = os.listdir(path)
        self.length = len(list_files)
        self.images = list(map(lambda _x: os.path.join(path, _x), list_files))

        path = os.path.join(self.path, 'masks')
        list_files = os.listdir(path)
        self.masks = list(map(lambda _x: os.path.join(path, _x), list_files))

    def __getitem__(self, item):
        path_img, path_mask = self.images[item], self.masks[item]
        img = Image.open(path_img).convert('RGB')
        mask = Image.open(path_mask).convert('L')

        img_np = np.array(img)
        mask_np = np.array(mask)

        if self.transform:
            augmented = self.transform(image=img_np, mask=mask_np)
            img = augmented['image']
            mask = augmented['mask']

            img = img.float() / 255.0
            mask = mask.float() / 255.0

            mask = (mask > 0.5).float()

        return img, mask

    def __len__(self):
        return self.length


class UNetModel(nn.Module):
    class _TwoConvLayers(nn.Module):
        def __init__(self, in_channels, out_channels, dropout=0.2):
            super().__init__()

            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(dropout),

                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(dropout)
            )

        def forward(self, x):
            return self.model(x)

    class _EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.block(x)
            y = self.max_pool(x)
            return y, x

    class _DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)

        def forward(self, x, y):
            x = self.transpose(x)
            u = torch.cat([x, y], dim=1)
            u = self.block(u)
            return u

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc_block1 = self._EncoderBlock(in_channels, 64)
        self.enc_block2 = self._EncoderBlock(64, 128)
        self.enc_block3 = self._EncoderBlock(128, 256)
        self.enc_block4 = self._EncoderBlock(256, 512)

        self.bottleneck = self._TwoConvLayers(512, 1024)

        self.dec_block1 = self._DecoderBlock(1024, 512)
        self.dec_block2 = self._DecoderBlock(512, 256)
        self.dec_block3 = self._DecoderBlock(256, 128)
        self.dec_block4 = self._DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        x, y4 = self.enc_block4(x)

        x = self.bottleneck(x)

        x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)

        return self.out(x)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = nn.functional.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2 * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


train_transforms = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.ElasticTransform(p=0.2, alpha=50, sigma=5),
        A.OpticalDistortion(p=0.2, distort_limit=0.05),
        A.GridDistortion(p=0.2, num_steps=3)
    ]),
    A.OneOf([
        A.Rotate(limit=15, p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.7,
            border_mode=cv2.BORDER_REFLECT
        )
    ]),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    ToTensorV2()
], additional_targets={'mask': 'image'})

test_transforms = A.Compose([
    A.Resize(512, 512),
    ToTensorV2()
], additional_targets={'mask': 'image'})

inference_transforms = A.Compose([
    A.Resize(512, 512),
    ToTensorV2()
])

d_train_full = SegmentDataset(r"dataset_seg", transform=train_transforms)
d_val_full = SegmentDataset(r"dataset_seg", transform=test_transforms)

indices = list(range(len(d_train_full)))
np.random.shuffle(indices)
split_inx = int(0.8 * len(indices))

train_dataset = data.Subset(d_train_full, indices[:split_inx])
val_dataset = data.Subset(d_val_full, indices[split_inx:])

train_data = data.DataLoader(train_dataset, 2, shuffle=True)
val_data = data.DataLoader(val_dataset, 2, shuffle=False)

model = UNetModel()

optimizer = optim.RMSprop(params=model.parameters(), lr=0.001)
loss_1 = nn.BCEWithLogitsLoss()
loss_2 = SoftDiceLoss()

epochs = 40
loss_lst = []
loss_lst_val = []

for _e in range(epochs):
    model.train()
    loss_mean = 1
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=False)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_1(predict, y_train) + loss_2(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
        train_tqdm.set_description(f"Train: Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.4f}")

    model.eval()
    Q_val = 0
    count_val = 0

    for x_val, y_val in val_data:
        with torch.no_grad():
            predict = model(x_val)
            loss = loss_1(predict, y_val) + loss_2(predict, y_val)

            count_val += 1
            Q_val += loss.item()

        Q_val /= count_val

    loss_lst.append(loss_mean)
    loss_lst_val.append(Q_val)

    print(f" | loss_mean={loss_mean:.3f}, Q_val={Q_val:.3f}")

st = model.state_dict()
torch.save(st, 'model_unet_seg.tar')

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_lst)+1), loss_lst, 'b-', label='Train Loss', linewidth=2)
plt.plot(range(1, len(loss_lst_val)+1), loss_lst_val, 'r-', label='Validation Loss', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

model.eval()

img = Image.open(r"car_1.jpg").convert('RGB')
img_np = np.array(img)

transformed = inference_transforms(image=img_np)
img = transformed['image']
img = img.float() / 255.0

img = img.unsqueeze(0)

with torch.no_grad():
    p = model(img).squeeze(0)

x = nn.functional.sigmoid(p.permute(1, 2, 0))
x = x.detach().numpy() * 255
x = np.clip(x, 0, 255).astype('uint8')
plt.imshow(x, cmap='gray')
plt.show()

