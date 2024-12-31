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
        if random.random() < 0.5:
            scale_factor = random.uniform(0.3, 1.2)
            image = scale_with_top_alignment(image, scale_factor)
            mask = scale_with_top_alignment(mask, scale_factor)
            spacing = spacing/scale_factor
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
        attn = self.sigmoid(self.conv(x + spacing_position))  # (B, 1, H, W)
        # attn = self.sigmoid(self.conv(attn))  # (B, 1, H, W)
        return x * attn  # 广播相乘

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, positional_encoding):
        """
        x: 输入张量，形状为 (num_patches, B, patch_h, patch_w)
        positional_encoding: 位置编码，形状为 (num_patches, B, patch_h, patch_w)
        """
        # 获取输入形状
        num_patches, B, patch_h, patch_w = x.size()

        # 确保连续后再调整形状
        x = x.contiguous().view(num_patches, B, -1)  # 转为 3D (sequence_length, batch_size, embed_dim)
        positional_encoding = positional_encoding.contiguous().view(num_patches, B, -1)

        # 加入位置编码
        x = x + positional_encoding

        # 多头自注意力
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)

        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # 恢复到原始形状
        x = x.view(num_patches, B, patch_h, patch_w)
        return x

class AttentionLayerWithTransformer(nn.Module):
    def __init__(self, input_channels, output_channels, num_heads=4, patch_size=16):
        super(AttentionLayerWithTransformer, self).__init__()
        self.patch_size = patch_size
        self.spatial_attention = SpatialAttention()  # 空间注意力模块

        # # Transformer Block
        # self.transformer = TransformerBlock(
        #     embed_dim=input_channels * (patch_size ** 2),
        #     num_heads=num_heads,
        #     ff_dim=4 * input_channels * (patch_size ** 2)
        # )

        # 通道压缩
        self.channel_reduction = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x, spacing_position):
        B, C, H, W = x.size()
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Height and Width must be divisible by patch_size"
        
        # 空间注意力
        x = self.spatial_attention(x, spacing_position)

        # # 分块操作
        # patch_h = H // self.patch_size
        # patch_w = W // self.patch_size
        # patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches = patches.contiguous().view(B, C, patch_h * patch_w, -1).permute(2, 0, 3, 1)  # (num_patches, B, patch_size**2 * C)

        # # 加入位置编码并通过 Transformer
        # spacing_position = spacing_position.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # spacing_position = spacing_position.contiguous().view(B, C, patch_h * patch_w, -1).permute(2, 0, 3, 1)  # (num_patches, B, patch_size**2 * C)

        # patches = self.transformer(patches, spacing_position)

        # # 重构回原始形状
        # patches = patches.permute(1, 3, 0, 2).contiguous()  # (B, C, num_patches, patch_size**2)
        # patches = patches.view(B, C, patch_h, patch_w, self.patch_size, self.patch_size)
        # x = patches.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)

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
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='mobilenet_v2', type=str, help='type of model, e.g., mobilenet_v2, vgg13...')
    args = parser.parse_args()
    encoder = args.encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("./test_result_depthawarev3.csv"):
        test_image_paths = getallfilesofwalk("/home/joe/Project/USSkin/SegData/images/test")
        test_mask_paths = [f.replace("images", "msks") for f in test_image_paths]
        df = pd.DataFrame({"image": test_image_paths, "mask": test_mask_paths})
    else:
        df = pd.read_csv("./test_result_depthawarev3.csv")

    # # 初始化模型、损失函数和优化器
    model = SegModel(encoder)

    model.to(device)
    model.load_state_dict(torch.load('./ckpt/unet++_5cls_{}_depthawarev3_2.pth'.format(encoder),weights_only=True))
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    height = 256
    width = 256
    channels = 16
    depth_encoding = positional_encoding_2d(height, width, channels).transpose( 1, 2, 0)    
    spacing_dict = {}
    df_spacing = pd.read_csv('/home/joe/Project/USSkin/spacing.csv')
    for i in range(len(df_spacing)):
        spacing_dict[df_spacing['filename'][i]] = df_spacing['spacing'][i]
    model.eval()
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
            adjusted_depth_encoding = get_real_position_encoding(depth_encoding, image.shape[0], image.shape[1], spacing)
            image = np.expand_dims(image, axis=-1)
            new_mask = []
            for j in range(5):
                mask_ = np.zeros((mask.shape[0], mask.shape[1]))
                mask_[np.where(mask == j)] = 1
                new_mask.append(mask_)
            mask = np.array(new_mask).transpose(1, 2, 0).astype(np.uint8)
            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            adjusted_depth_encoding = cv2.resize(adjusted_depth_encoding, (256, 256))
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()/255
            mask = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).float()
            adjusted_depth_encoding = torch.from_numpy(adjusted_depth_encoding).permute(2, 0, 1).unsqueeze(0).float()
                # print(image.shape)

            images = image.to(device)
            masks = mask.to(device)
            adjusted_depth_encoding = adjusted_depth_encoding.to(device)

            # 前向传播
            outputs = model(images, adjusted_depth_encoding)
            
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
    df.to_csv('./test_result_depthawarev3.csv', index=False)
