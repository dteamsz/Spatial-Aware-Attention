# created by Joe Tao
# Date: 20201009
import os

import numpy as np
import cv2
# from skimage.transform import resize
def write_avi(video_data, output_path, fps=30, codec='XVID'):
    """
    将 ndarray 保存为视频文件。

    参数：
    - video_data: numpy.ndarray, 形状为 (num_frames, height, width, channels) 或 (num_frames, height, width)。
                  如果是灰度图像（单通道），函数会自动转换为 BGR 格式。
    - output_path: str, 视频保存路径（包含扩展名）。
    - fps: int, 视频帧率，默认为 30。
    - codec: str, 视频编码格式，默认为 'XVID'。

    返回：
    - None
    """
    # 确保 video_data 是 numpy 数组
    if not isinstance(video_data, np.ndarray):
        raise ValueError("video_data 必须是 numpy.ndarray 类型")

    # 获取视频维度
    num_frames, height, width = video_data.shape[:3]
    channels = 1 if len(video_data.shape) == 3 else video_data.shape[3]
    
    if channels == 1:  # 如果是灰度图像，将其转换为 BGR
        video_data = np.stack([video_data] * 3, axis=-1)
    
    # 定义视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 写入每一帧
    for frame in video_data:
        video_writer.write(frame)
    
    # 释放资源
    video_writer.release()
    print(f"视频已保存到 {output_path}")

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


def create_dir(dir_name):
    try:
        # Create target Directory
        os.makedirs(dir_name)
    except FileExistsError:
        print("Directory ", dir_name, " already exists")


# def zero_pad(img, size=448):
#     '''
#     pad zeros to make a square img for resize
#     '''
#     h, w, c = img.shape
#     if h > w:
#         zeros = np.zeros([h, h - w, c]).astype(np.uint8)
#         img_padded = np.hstack((img, zeros))
#     elif h < w:
#         zeros = np.zeros([w - h, w, c]).astype(np.uint8)
#         img_padded = np.vstack((img, zeros))
#     else:
#         img_padded = img
#
#     img_resized = (255*resize(img_padded, (size, size), anti_aliasing=True)).astype(np.uint8)
#
#     return img_resized


def read_avi(fname):
    cap = cv2.VideoCapture(fname)

    wid = int(cap.get(3))
    hei = int(cap.get(4))
    framerate = int(cap.get(5))
    framenum = int(cap.get(7))

    video = np.zeros((framenum, hei, wid, 3), dtype=np.uint8)

    for i in range(framenum):
        a, b = cap.read()
        video[i] = b[..., ::-1]
    for i in range(framenum):
        video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
    spacing = None
    fps = framerate
    return video, spacing, fps

def letterbox_image(image_src, dst_size=(512,512), pad_color=(0)):
    """
    缩放图片，保持长宽比。
    :param image_src:       原图（numpy）
    :param dst_size:        （h，w）
    :param pad_color:       填充颜色，默认是灰色
    :return:
    """
    src_h, src_w = image_src.shape[:2]
    dst_h, dst_w = dst_size
    scale = min(dst_h / src_h, dst_w / src_w)
    pad_h, pad_w = int(round(src_h * scale)), int(round(src_w * scale))

    if image_src.shape[0:2] != (pad_w, pad_h):
        image_dst = cv2.resize(image_src, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
    else:
        image_dst = image_src

    top = int((dst_h - pad_h) / 2)
    down = int((dst_h - pad_h + 1) / 2)
    left = int((dst_w - pad_w) / 2)
    right = int((dst_w - pad_w + 1) / 2)

    # add border
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    x_offset, y_offset = max(left, right) / dst_w, max(top, down) / dst_h
    return image_dst, x_offset, y_offset 