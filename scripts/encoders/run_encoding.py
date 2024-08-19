from torchvision import models, transforms
from PIL import Image, ImageOps
import sys
from tqdm import tqdm
import torch
import os
import numpy as np 
from transformers import CLIPProcessor, CLIPModel



from pdb import set_trace as pause

encoder_name = sys.argv[3]


if encoder_name == 'clip':
    encoding_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    encoding_model.to('cuda')
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")


if encoder_name == 'dino':
    encoding_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    encoding_model.to(device)


if encoder_name == 'resnet':
    encoding_model = models.resnet152(pretrained=True)
    encoding_model = torch.nn.Sequential(*list(encoding_model.children())[:-1])
    encoding_model.eval()

# CLIP


def run_clip(img):
    img = Image.open(img)
    img = processor(images=img, return_tensors='pt')["pixel_values"].to('cuda')
    
    embedding = encoding_model.get_image_features(img).cpu().detach().numpy()

    return embedding


# Normalize for dino
transform_dino = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize(244), 
    transforms.CenterCrop(224),
    transforms.Normalize([0.5], [0.5])])

def preprocess_dino(img):
    transformed_img = transform_dino(img)[:3].unsqueeze(0)

    return transformed_img

def run_dino(img):
    img = Image.open(img)
    img = preprocess_dino(img)

    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        encoding = encoding_model(img.to(device))
        encoding = encoding.cpu().numpy()

    
    return encoding

# Normalize for resnet
transform_resnet = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

# Pre-process for resnet
def preprocess_resnet(img):
    target_size = max(img.size)
    img = ImageOps.pad(img, (target_size, target_size))
    img = img.resize((224,224))
    
    return transform_resnet(img).unsqueeze(0)


def run_resnet(img_path):
    img = Image.open(img_path)
    tensor = preprocess_resnet(img)

    encoding = encoding_model(tensor)
    encoding = encoding.detach().numpy().reshape(-1, np.prod(encoding.size()[1:]))
    
    return encoding


def run_encoder(img_path, out_dir, dataset_dir):
    img_name = os.path.basename(img_path).replace('.png','')
    img_dir = os.path.dirname(img_path)

    result_dir = img_dir.replace(dataset_dir, out_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, img_name + '.npy')

    if encoder_name == 'clip':
        encoding = run_clip(img_path)
    if encoder_name == 'resnet':
        encoding = run_resnet(img_path)
    if encoder_name == 'dino':
        encoding = run_dino(img_path)

    np.save(result_path, encoding)

dataset_dir = sys.argv[1]
out_dir = sys.argv[2]

img_folders = []
for dirpaths, dirnames, filenames in os.walk(dataset_dir):
    if not dirnames: 
        img_folders += [dirpaths]

print('running with encoder: {}'.format(encoder_name))

for img_folder in img_folders:
    img_files = os.listdir(img_folder)
    print(img_folder)
    for img_file in tqdm(img_files):
        img_path = os.path.join(img_folder, img_file)
        run_encoder(img_path, out_dir, dataset_dir)

