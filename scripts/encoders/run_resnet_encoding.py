from torchvision import models, transforms
from PIL import Image, ImageOps
import sys
from tqdm import tqdm
import torch
import os
import numpy as np 

encoding_model = models.resnet152(pretrained=True)
encoding_model = torch.nn.Sequential(*list(encoding_model.children())[:-1])
encoding_model.eval()

# Normalize for resnet
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

# Pre-process for resnet
def preprocess_resnet(img):
    target_size = max(img.size)
    img = ImageOps.pad(img, (target_size, target_size))
    img = img.resize((224,224))
    
    return transform(img).unsqueeze(0)

def run_resnet(img_path, out_dir, dataset_dir):
    img_name = os.path.basename(img_path).replace('.png','')
    img_dir = os.path.dirname(img_path)

    result_dir = img_dir.replace(dataset_dir, out_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, img_name + '.npy')

    img = Image.open(img_path)
    tensor = preprocess_resnet(img)

    encoding = encoding_model(tensor)
    encoding = encoding.detach().numpy().reshape(-1, np.prod(encoding.size()[1:]))
    
    np.save(result_path, encoding)

dataset_dir = sys.argv[1]
out_dir = sys.argv[2]

img_folders = []
for dirpaths, dirnames, filenames in os.walk(dataset_dir):
    if not dirnames: 
        img_folders += [dirpaths]


for img_folder in img_folders:
    img_files = os.listdir(img_folder)
    print(img_folder)
    for img_file in tqdm(img_files):
        img_path = os.path.join(img_folder, img_file)
        run_resnet(img_path, out_dir, dataset_dir)

