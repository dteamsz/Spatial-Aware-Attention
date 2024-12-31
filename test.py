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
    
class SegmentationDataset(Dataset):
    def __init__(self, df, transform=False):
        self.df = df
        self.transform = transform
        self.augmenter = iaa.Sequential([
                    iaa.SomeOf((1, 3), [  # 从以下增强器中随机选择 1 到 3 个
                        iaa.Fliplr(0.5),
                        iaa.GaussianBlur(sigma=(0, 3.0)),
                        iaa.AdditiveGaussianNoise(scale=(10, 50)),
                        iaa.Affine(translate_percent=(-0.3, 0.3)),
                        iaa.Affine(scale=(0.3, 1))
                    ]),
                        iaa.Multiply((0.8, 1.2))  # 再随机调整亮度
                    ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # try:
        image = cv2.imread(self.df['image'][idx], cv2.IMREAD_GRAYSCALE)
        image = image.reshape((1, image.shape[0], image.shape[1], 1))
        mask = cv2.imread(self.df['mask'][idx], cv2.IMREAD_GRAYSCALE)
        new_mask = []
        for j in range(4):
            mask_ = np.zeros((mask.shape[0], mask.shape[1]))
            mask_[np.where(mask == j)] = 1
            new_mask.append(mask_)
        mask = np.array(new_mask).reshape((1, mask.shape[0], mask.shape[1], 4)).astype(np.uint8)

        if self.transform:
            augmented_images = self.augmenter(images=image, segmentation_maps=mask)
            image, mask = augmented_images
        image = cv2.resize(image[0,:,:,0], (256, 256))
        mask = cv2.resize(mask[0,:,:], (256, 256))
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
    
if __name__ == "__main__":
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='mobilenet_v2', type=str, help='type of model, e.g., mobilenet_v2, vgg13...')
    args = parser.parse_args()
    encoder = args.encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("./test_result.csv"):
        test_image_paths = getallfilesofwalk("/home/joe/Project/USSkin/SegData/images/test")
        test_mask_paths = [f.replace("images", "msks") for f in test_image_paths]
        df = pd.DataFrame({"image": test_image_paths, "mask": test_mask_paths})
    else:
        df = pd.read_csv("./test_result.csv")

    # # 初始化模型、损失函数和优化器
    model = smp.UnetPlusPlus(
        encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=5  ,                      # model output channels (number of classes in your dataset)
    )
    # print(model)
    model.to(device)
    model.load_state_dict(torch.load('./ckpt/unet++_5cls_{}.pth'.format(encoder),weights_only=True))
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    model.eval()
    spacing_dict = {}
    df_spacing = pd.read_csv('/home/joe/Project/USSkin/spacing.csv')
    for i in range(len(df_spacing)):
        spacing_dict[df_spacing['filename'][i]] = df_spacing['spacing'][i]
    with torch.no_grad():
        for i in tqdm(range(len(df))):
            image = cv2.imread(df['image'][i], cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(df['mask'][i], cv2.IMREAD_GRAYSCALE)
            spacing = spacing_dict[df['image'][i].split('/')[-1]]
            # if random.random() < 0.5:
            #     scale_factor = random.uniform(0.3, 0.5)
            #     image = scale_with_top_alignment(image, scale_factor)
            #     mask = scale_with_top_alignment(mask, scale_factor)
            #     spacing = spacing/scale_factor
            new_mask = []
            for j in range(5):
                mask_ = np.zeros((mask.shape[0], mask.shape[1]))
                mask_[np.where(mask == j)] = 1
                new_mask.append(mask_)
            mask = np.array(new_mask).transpose(1, 2, 0).astype(np.uint8)
            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)/255
            mask = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).float()
                # print(image.shape)

            images = image.to(device)
            masks = mask.to(device)

            # 前向传播
            outputs = model(images)
            
            # 计算混合损失 (BCE Loss + Dice Loss)
            dice_up = 1-dice_loss(outputs[:,0,:,:].unsqueeze(1), masks[:,0,:,:].unsqueeze(1)).item()
            dice_epi = 1-dice_loss(outputs[:,1,:,:].unsqueeze(1), masks[:,1,:,:].unsqueeze(1)).item()
            dice_der = 1-dice_loss(outputs[:,2,:,:].unsqueeze(1), masks[:,2,:,:].unsqueeze(1)).item()
            dice_sub = 1-dice_loss(outputs[:,3,:,:].unsqueeze(1), masks[:,3,:,:].unsqueeze(1)).item() 
            dice_bottom = 1-dice_loss(outputs[:,4,:,:].unsqueeze(1), masks[:,4,:,:].unsqueeze(1)).item()
            df.loc[i, 'dice_up_{}'.format(encoder)] = dice_up
            df.loc[i, 'dice_epi_{}'.format(encoder)] = dice_epi
            df.loc[i, 'dice_der_{}'.format(encoder)] = dice_der
            df.loc[i, 'dice_sub_{}'.format(encoder)] = dice_sub
            df.loc[i, 'dice_bottom_{}'.format(encoder)] = dice_bottom
    print('mean_dice_up_{}'.format(encoder), df['dice_up_{}'.format(encoder)].mean())
    print('mean_dice_epi_{}'.format(encoder), df['dice_epi_{}'.format(encoder)].mean())
    print('mean_dice_der_{}'.format(encoder), df['dice_der_{}'.format(encoder)].mean())
    print('mean_dice_sub_{}'.format(encoder), df['dice_sub_{}'.format(encoder)].mean())
    print('mean_dice_bottom_{}'.format(encoder), df['dice_bottom_{}'.format(encoder)].mean())
    print('mean_dice_{}'.format(encoder), sum(df['dice_up_{}'.format(encoder)]+df['dice_epi_{}'.format(encoder)]+df['dice_der_{}'.format(encoder)]+df['dice_sub_{}'.format(encoder)]+df['dice_bottom_{}'.format(encoder)])/(5*len(df)))

    df.to_csv('./test_result.csv', index=False)
