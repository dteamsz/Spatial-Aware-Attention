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
import random

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

def positional_encoding_2d(height, width, channels):
    """
    使用 NumPy 生成二维三角位置编码
    :param height: 特征图的高度
    :param width: 特征图的宽度
    :param channels: 编码的通道数（必须是4的倍数）
    :return: 位置编码矩阵，形状为 (channels, height, width)
    """
    assert channels % 4 == 0, "The number of channels must be a multiple of 4"
    
    c_div_4 = channels // 4

    # 生成位置矩阵
    y_pos = np.arange(height)[:, np.newaxis]  # 高度方向的索引 (H, 1)
    x_pos = np.arange(width)[np.newaxis, :]  # 宽度方向的索引 (1, W)

    # 计算缩放因子
    div_term = np.exp(np.arange(c_div_4) * -(np.log(10000.0) / c_div_4))  # (C/4,)

    # 初始化位置编码
    pos_encoding = np.zeros((channels, height, width), dtype=np.float32)

    # 计算正弦和余弦位置编码
    pos_encoding[0:c_div_4, :, :] = np.sin(y_pos * div_term[:, np.newaxis, np.newaxis])  # (C/4, H, W)
    pos_encoding[c_div_4:2 * c_div_4, :, :] = np.cos(y_pos * div_term[:, np.newaxis, np.newaxis])  # (C/4, H, W)
    pos_encoding[2 * c_div_4:3 * c_div_4, :, :] = np.sin(x_pos * div_term[:, np.newaxis, np.newaxis])  # (C/4, H, W)
    pos_encoding[3 * c_div_4:, :, :] = np.cos(x_pos * div_term[:, np.newaxis, np.newaxis])  # (C/4, H, W)

    return pos_encoding

def ultrasound_spacing_encoding(depth_range, depth_dim, width_dim):
    """
    生成超声深度和宽度的二维位置编码
    :param depth_range: 最大深度（int）
    :param width_range: 最大宽度（int）
    :param depth_dim: 深度编码维度
    :param width_dim: 宽度编码维度
    :return: 编码矩阵 [depth_range, width_range, depth_dim + width_dim]
    """
    # 初始化位置矩阵
    depth_positions = np.arange(depth_range)[:, np.newaxis]  # 深度 [depth_range, 1]
    
    # 深度编码
    depth_div_term = np.exp(np.arange(0, depth_dim, 2) * -(np.log(10000.0) / depth_dim))
    depth_encoding = np.zeros((depth_range, depth_dim))
    depth_encoding[:, 0::2] = np.sin(depth_positions * depth_div_term)  # 偶数维
    depth_encoding[:, 1::2] = np.cos(depth_positions * depth_div_term)  # 奇数维

    
    return depth_encoding

def get_real_position_encoding(depth_encoding,  depth, width,  spacing):

    depth_factor = depth*spacing/5
    # width_factor = width*spacing/5
    # print(depth_factor, width_factor)
    depth_encoding = depth_encoding[:int(depth_factor*depth_encoding.shape[1]),:]
    depth_encoding = cv2.resize(depth_encoding, (width, depth), interpolation=cv2.INTER_LINEAR)

    return depth_encoding

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
        self.df_spacing = pd.read_csv('/home/joe/Project/USSkin/spacing.csv')
        # print(self.df_spacing.columns)
        # self.depth_encoding = ultrasound_spacing_encoding(depth_range=256,  depth_dim=16)
        height = 256
        width = 256
        channels = 16

        # 初始化位置编码模块
        self.depth_encoding = positional_encoding_2d(height, width, channels).transpose( 1, 2, 0)
        print(self.depth_encoding.shape)
        self.spacing_dict = {}
        for i in range(len(self.df_spacing)):
            self.spacing_dict[self.df_spacing['filename'][i]] = self.df_spacing['spacing'][i]
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # try:

        image = cv2.imread(self.df['image'][idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.df['mask'][idx], cv2.IMREAD_GRAYSCALE)
        spacing = self.spacing_dict[self.df['image'][idx].split('/')[-1]]
        # if random.random() < 0.5:
        #     scale_factor = random.uniform(0.3, 1.2)
        #     image = scale_with_top_alignment(image, scale_factor)
        #     mask = scale_with_top_alignment(mask, scale_factor)
        #     spacing = spacing/scale_factor
        # print('s:',spacing)
        adjusted_depth_encoding = get_real_position_encoding(self.depth_encoding, image.shape[0], image.shape[1], spacing)
        adjusted_depth_encoding = adjusted_depth_encoding.reshape((1, adjusted_depth_encoding.shape[0], adjusted_depth_encoding.shape[1], 16))
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
        # print(image.shape, adjusted_depth_encoding.shape)
        image = cv2.resize(image[0,:,:,:], (256, 256))
        adjusted_depth_encoding = cv2.resize(adjusted_depth_encoding[0,:,:,:], (256, 256))
        mask = cv2.resize(mask[0,:,:,:], (256, 256))
        image = torch.from_numpy(image).unsqueeze(-1).permute(2, 0, 1).float()/255
        adjusted_depth_encoding = torch.from_numpy(adjusted_depth_encoding).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        # print(image.shape)
        # print(mask.shape)
        # except:
        #     print(self.df['image'][idx])
        return image, mask, adjusted_depth_encoding

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

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 空间注意力模块
        self.conv = nn.Conv2d(16, 1, kernel_size=7, padding=3, bias=False)  # 7x7卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, spacing_position):
        # x: (B, C, H, W)
        # avg_pool = torch.mean(spacing_position, dim=1, keepdim=True)  # (B, 1, H, W)
        # max_pool, _ = torch.max(spacing_position, dim=1, keepdim=True)  # (B, 1, H, W)
        # attn = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        attn = self.sigmoid(self.conv(x+spacing_position))  # (B, 1, H, W)
        # attn = self.sigmoid(self.conv(attn))  # (B, 1, H, W)
        return x * attn  # 广播相乘

class AttentionLayerWithTransformer(nn.Module):
    def __init__(self, input_channels, output_channels, num_heads=4, patch_size=16):
        super(AttentionLayerWithTransformer, self).__init__()
        self.patch_size = patch_size
        self.spatial_attention = SpatialAttention()  # 空间注意力模块
        # 通道压缩
        self.channel_reduction = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x, spacing_position):
        # 空间注意力
        x = self.spatial_attention(x, spacing_position)
        # 通道压缩
        x = self.channel_reduction(x)
        return x



class SegModel(nn.Module):
    def __init__(self,encoder):
        super(SegModel, self).__init__()
        self.base = smp.UnetPlusPlus(
            encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=5,                      # model output channels (number of classes in your dataset)
            )
        self.postional_attention =  AttentionLayerWithTransformer(16, 16)
    def forward(self, x, spacing_position):
        features, decoder_output = self.base(x, return_features=True)
        x = self.postional_attention(decoder_output, spacing_position)
        x = self.base.segmentation_head(x)

        return x

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
    model = SegModel(encoder)
    model.load_state_dict(torch.load("./ckpt/unet++_5cls_{}_depthawarev3_2.pth".format(encoder)))
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
        for images, masks, spacing_position in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            spacing_position = spacing_position.to(device)
            # mask_ = masks[0].cpu().numpy()
            # print(mask_.shape)
            # mask_ = np.argmax(mask_, axis=0)*50
            # print(np.max(mask_))
            # cv2.imwrite('./msk.jpg',mask_)
            # 前向传播
            outputs = model(images, spacing_position)
            
            # 计算混合损失 (BCE Loss + Dice Loss)
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice_loss += dice_loss(outputs, masks).item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Dice Loss: {epoch_dice_loss/len(train_loader):.4f}")
        with open("./log/unet++_5cls_{}_depthawarev3.log".format(encoder), "a") as f:
            f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Dice Loss: {epoch_dice_loss/len(train_loader):.4f}\n")
        
        model.eval()
        epoch_loss = 0
        epoch_dice_loss = 0
        with torch.no_grad():
            for images, masks, spacing_position in tqdm(val_loader):
                images = images.to(device)
                masks = masks.to(device)
                spacing_position = spacing_position.to(device)
                outputs = model(images, spacing_position)
                loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
                epoch_loss += loss.item()
                epoch_dice_loss += dice_loss(outputs, masks).item()
            print(f"Validation Loss: {epoch_loss/len(val_loader):.4f}, Dice Loss: {epoch_dice_loss/len(val_loader):.4f}")
        
        with open("./log/unet++_5cls_{}_depthawarev3.log".format(encoder), "a") as f:
            f.write(f"Validation Loss: {epoch_loss/len(val_loader):.4f}, Dice Loss: {epoch_dice_loss/len(val_loader):.4f}\n")
        
        if epoch_loss < best_val_loss:
            # 保存模型
            torch.save(model.state_dict(), "./ckpt/unet++_5cls_{}_depthawarev3_2.pth".format(encoder))
            best_val_loss = epoch_loss
            print("模型训练完成并已保存！")

