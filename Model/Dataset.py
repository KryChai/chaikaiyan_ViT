# # 导入库
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import optim
# from torch.utils.data import Dataset, DataLoader, random_split
# import warnings
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# warnings.filterwarnings("ignore")
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2 # 用于图像增强和转换为PyTorch张量。
#
# torch.manual_seed(17)
#
# # 自定义数据集CamVidDataset
# class CamVidDataset(Dataset):
#     def __init__(self, images_dir, masks_dir):
#         self.transform = A.Compose([
#             A.Resize(512, 512),  # 调整图像大小
#             A.HorizontalFlip(),  # 水平翻转
#             A.VerticalFlip(),  # 垂直翻转
#             A.Normalize(),  # 归一化
#             ToTensorV2(),  # 转换为PyTorch张量
#         ])
#         self.ids = os.listdir(images_dir)  # 获取图像文件名列表
#         self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]  # 构建图像文件路径列表
#         self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]  # 构建mask文件路径列表
#
#     def __getitem__(self, i):
#         # 读取图像和mask数据
#         image = np.array(Image.open(self.images_fps[i]).convert('L'))  # 将图像转换为灰度图
#         mask = np.array(Image.open(self.masks_fps[i]).convert('L'))  # 将mask转换为灰度图
#         image = self.transform(image=image, mask=mask)  # 应用图像变换
#
#         # 确保 image['image'] 是一个三维张量
#         image_tensor = image['image'].unsqueeze(0) if image['image'].ndim == 2 else image['image']
#
#         # 确保 image['mask'] 是一个三维张量
#         mask_tensor = image['mask'][:, :, 0].unsqueeze(0) if image['mask'].ndim == 2 else image['mask']
#
#         return image_tensor, mask_tensor
#
#     def __len__(self):
#         return len(self.ids)  # 返回数据集的大小
#
# # 设置数据集路径
# DATA_DIR_TRAIN = r'/tmp/XCAD/train'  # 根据自己的路径来设置
# x_train_dir = os.path.join(DATA_DIR_TRAIN, 'images')
# y_train_dir = os.path.join(DATA_DIR_TRAIN, 'masks')
#
# DATA_DIR_TEST = r'/tmp/XCAD/test'
# x_valid_dir = os.path.join(DATA_DIR_TRAIN, 'images')
# y_valid_dir = os.path.join(DATA_DIR_TRAIN, 'masks')
#
# # 创建训练和验证数据集
# train_dataset = CamVidDataset(
#     x_train_dir,
#     y_train_dir,
# )
# val_dataset = CamVidDataset(
#     x_valid_dir,
#     y_valid_dir,
# )
#
# print(len(train_dataset))
# print(len(val_dataset))
#
# # 创建数据加载器
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, drop_last=True)
#
#

# 导入库
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

torch.manual_seed(17)

# 自定义数据集CamVidDataset
# XCAD
class CamVidDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir):
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ])
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

    def __getitem__(self, i):
        # read data
        image = np.array(Image.open(self.images_fps[i]).convert('L'))
        mask = np.array(Image.open(self.masks_fps[i]).convert('L'))
        image = self.transform(image=image, mask=mask)

        # return image['image'], image['mask'][:, :, 0]
        return image['image'], image['mask']

    def __len__(self):
        return len(self.ids)


# 设置数据集路径
DATA_DIR_TRAIN = r'/tmp/XCAD/train'  # 根据自己的路径来设置
x_train_dir = os.path.join(DATA_DIR_TRAIN, 'images')
y_train_dir = os.path.join(DATA_DIR_TRAIN, 'masks')

DATA_DIR_TEST = r'/tmp/XCAD/test'
x_valid_dir = os.path.join(DATA_DIR_TRAIN, 'images')
y_valid_dir = os.path.join(DATA_DIR_TRAIN, 'masks')

# 创建训练和验证数据集
train_dataset = CamVidDataset(
    x_train_dir,
    y_train_dir,
)
val_dataset = CamVidDataset(
    x_valid_dir,
    y_valid_dir,
)

print(len(train_dataset))
print(len(val_dataset))

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, drop_last=True)
