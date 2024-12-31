import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import os
import imgaug.augmenters as iaa
import cv2
import pandas as pd
import argparse


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, targets):
        smooth = 1.0
        outputs = torch.sigmoid(outputs)
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        intersection = (outputs * targets).sum()
        dice = (2. * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)
        return 1 - dice
def scale_with_top_alignment(image, scale):
    # 获取原始图像尺寸
    height, width = image.shape[:2]
    
    # 计算缩放后的尺寸
    new_height, new_width = int(height * scale), int(width * scale)
    
    # 缩放图像
    resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    if scale > 1.0:
        # 放大：裁剪顶部区域
        cropped_img = resized_img[:height, :width]  # 裁剪到原始大小
        return cropped_img
    else:
        # 缩小：填充底部
        padded_img = np.zeros_like(image)  # 创建黑色背景
        padded_img[:new_height, :new_width] = resized_img  # 将缩小图像放置顶部
        return padded_img
class SegmentationDataset(Dataset):
    def __init__(self, df, transform=False):
        self.df = df
        self.transform = transform
        self.augmenter = iaa.Sequential([
                    iaa.SomeOf((0, 3), [  # 从以下增强器中随机选择 1 到 3 个
                        iaa.Fliplr(0.5),
                        iaa.GaussianBlur(sigma=(0, 3.0)),
                        iaa.AdditiveGaussianNoise(scale=(10, 50)),
                        iaa.Affine(translate_percent=(-0.3, 0.3)),
                    ]),
                        iaa.Multiply((0.8, 1.2))  # 再随机调整亮度
                    ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # try:
        image = cv2.imread(self.df['image'][idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.df['mask'][idx], cv2.IMREAD_GRAYSCALE)
        # if random.random() < 0.5:
        #     scale_factor = random.uniform(0.3, 1.2)
        #     image = scale_with_top_alignment(image, scale_factor)
        #     mask = scale_with_top_alignment(mask, scale_factor)
        image = image.reshape((1, image.shape[0], image.shape[1], 1))
        # print(np.unique(mask))
        new_mask = []
        for j in range(5):
            mask_ = np.zeros((mask.shape[0], mask.shape[1]))
            mask_[np.where(mask == j)] = 1
            new_mask.append(mask_)
        mask = np.array(new_mask).transpose(1, 2, 0).astype(np.uint8)
        mask = np.expand_dims(mask, axis=0)
        if self.transform:
            augmented_images = self.augmenter(images=image, segmentation_maps=mask)
            image, mask = augmented_images
        image = cv2.resize(image[0,:,:,0], (256, 256))
        mask = cv2.resize(mask[0,:,:,:], (256, 256))
        image = torch.from_numpy(image).unsqueeze(0)/255
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        # print(image.shape)
        # print(mask.shape)
        # except:
        #     print(self.df['image'][idx])
        return image, mask

def getallfilesofwalk(root):
    """
    使用listdir循环遍历文件夹中所有文件
    """
    if not os.path.isdir(root):
        print(root)
        return []

    dirlist = os.walk(root)
    allfiles = []
    for root, dirs, files in dirlist:
        for file in files:
            #            print(os.path.join(root, file))
            allfiles.append(os.path.join(root, file))

    return allfiles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--encoder', default='mobilenet_v2', type=str, help='type of model, e.g., mobilenet_v2, vgg13...')
    args = parser.parse_args()
    encoder = args.encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001

    train_image_paths = getallfilesofwalk("/home/joe/Project/USSkin/SegData/images/train")
    train_mask_paths = [f.replace("images", "msks") for f in train_image_paths]
    df_train = pd.DataFrame({"image": train_image_paths, "mask": train_mask_paths})
    val_image_paths = getallfilesofwalk("/home/joe/Project/USSkin/SegData/images/val")
    val_mask_paths = [f.replace("images", "msks") for f in val_image_paths]
    df_val = pd.DataFrame({"image": val_image_paths, "mask": val_mask_paths})
    # 加载数据集
    train_dataset = SegmentationDataset(df_train, transform=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = SegmentationDataset(df_val, transform=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # # 初始化模型、损失函数和优化器
    model = smp.UnetPlusPlus(
        encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=5,                      # model output channels (number of classes in your dataset)
    )
    # model.load_state_dict(torch.load("./ckpt/unet++_5cls_{}.pth".format(encoder)))
    model.to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_loss = 1e100
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_dice_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            # mask_ = masks[0].cpu().numpy()
            # print(mask_.shape)
            # mask_ = np.argmax(mask_, axis=0)*50
            # print(np.max(mask_))
            # cv2.imwrite('./msk.jpg',mask_)
            # 前向传播
            outputs = model(images)
            
            # 计算混合损失 (BCE Loss + Dice Loss)
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice_loss += dice_loss(outputs, masks).item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Dice Loss: {epoch_dice_loss/len(train_loader):.4f}")
        model.eval()
        epoch_loss = 0
        epoch_dice_loss = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
                epoch_loss += loss.item()
                epoch_dice_loss += dice_loss(outputs, masks).item()
            print(f"Validation Loss: {epoch_loss/len(val_loader):.4f}, Dice Loss: {epoch_dice_loss/len(val_loader):.4f}")

        with open("./log/unet++_5cls_{}.log".format(encoder), "a") as f:
            f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Dice Loss: {epoch_dice_loss/len(train_loader):.4f}\n")
            f.write(f"Validation Loss: {epoch_loss/len(val_loader):.4f}, Dice Loss: {epoch_dice_loss/len(val_loader):.4f}\n")
        
        if epoch_loss < best_val_loss:
            # 保存模型
            torch.save(model.state_dict(), "./ckpt/unet++_5cls_{}.pth".format(encoder))
            best_val_loss = epoch_loss
            print("模型训练完成并已保存！")

