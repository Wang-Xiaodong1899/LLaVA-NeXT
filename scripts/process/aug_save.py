import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import GaussianBlur
import argparse
import random

# 定义数据增强变换
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 0.3)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    # transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
    transforms.RandomHorizontalFlip()
])

def augment_image(image, transform, state):
    # 使用指定的随机状态
    torch.set_rng_state(state)
    return transform(image)

def process_images(input_folder, output_folder, add_aug):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取文件夹中所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 生成一个随机状态用于所有图片
    initial_state = torch.get_rng_state()

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        
        # 保存原始图片
        # image.save(os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}_original.jpg'))
        
        # 如果需要增强
        if add_aug:
            # 使用相同的随机状态
            aug_image = augment_image(image, aug_transform, initial_state)
            aug_image.save(os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}_augmented.jpg'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Augmentation Script")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder with images.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder for processed images.')
    parser.add_argument('--add_aug', action='store_true', help='Whether to apply augmentation to images.')
    args = parser.parse_args()
    
    process_images(args.input_folder, args.output_folder, args.add_aug)
