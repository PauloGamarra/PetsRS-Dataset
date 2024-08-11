import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
import ultralytics
ultralytics.checks()
from PIL import Image, ImageOps
import numpy as np
import os
from tqdm import tqdm

from pdb import set_trace as pause

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
        
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

def detect_pet(img_pth, model):
    img = Image.open(img_pth)
    result = model.predict(img)
    result = result[0]

    # results are already sorted by confidence
    # so we get the first interest bounding box
    interest_box = None
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        if class_id in interest_classes:
            interest_box = box
            break
    if interest_box == None:
        return None

    cords = interest_box.xyxy[0].tolist()
    cords = [round(x) for x in cords]

    return cords

def segment_img(image, masks):   
    # Converta a máscara em uma matriz booleana
    mask = masks[0] > 0

    # Aplique a máscara à imagem
    segmented_image = np.zeros_like(image)
    for c in range(3):  # para cada canal de cor
        segmented_image[..., c] = image[..., c] * mask

    # Converta a imagem segmentada em uma imagem PIL e salve-a
    segmented_image_pil = Image.fromarray(segmented_image)

    # Converta para o modo RGB
    segmented_image_pil = segmented_image_pil.convert("RGB")

    return segmented_image_pil

def save_no_pet(img_file):
    with open('no_pet_imgs.txt', 'a') as file:
        no_pet_entry = '{}\n'.format(img_file)
        file.write(no_pet_entry)

# initialize yolo v8 for object detection
yolo_model = ultralytics.YOLO("yolov8x.pt")
yolo_model.to('cuda')

# initalize SAM model - metai ai for fast and precise segmentation
sam_checkpoint = "sam_weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


global num_vis_images

def yolo_sam(img_path, imgs_dir):
    global num_vis_images

    # Open image
    img_name = os.path.basename(img_path).replace('.png','')
    image_pil = Image.open(img_path)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # pre_process yolov8
    yolo_xyxy = detect_pet(img_path, yolo_model)

    if yolo_xyxy is None:
        print('no pet detected on {}'.format(img_path))
        save_no_pet(img_path)
        return


    
    predictor.set_image(image)

    # use xyxy object box from yolo
    input_box = np.array(yolo_xyxy)

    # pre_process and segment image object, removing background
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    print(os.path.dirname(img_path))
    print(imgs_dir)

    origin_species_dir = os.path.dirname(img_path).replace(imgs_dir,'')
    sam_vis_dir = os.path.join('sam_vis', origin_species_dir)
    if not os.path.exists(sam_vis_dir):
        os.makedirs(sam_vis_dir)

    if num_vis_images > 0:
        # print result
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')
        sam_vis_path = os.path.join(sam_vis_dir, '{}.png'.format(img_name))
        plt.savefig('{}_vis_sam.png'.format(sam_vis_path))
        plt.close()
        num_vis_images -= 1

    # save image result
    segmented_img = segment_img(image, masks)

    # crop 
    crop_segmented = segmented_img.crop(yolo_xyxy)
    crop_original = image_pil.crop(yolo_xyxy)

    crop_seg_dir = os.path.join('crop_segmented', origin_species_dir)
    crop_original_dir = os.path.join('crop_original', origin_species_dir)

    if not os.path.exists(crop_seg_dir):
        os.makedirs(crop_seg_dir)
    if not os.path.exists(crop_original_dir):
        os.makedirs(crop_original_dir)

    crop_seg_path = os.path.join(crop_seg_dir, '{}.png'.format(img_name))
    crop_original_path = os.path.join(crop_original_dir, '{}.png'.format(img_name))

    crop_segmented.save(crop_seg_path)
    crop_original.save(crop_original_path)



pets_imgs_dir = sys.argv[1]

lost_dir = os.path.join(pets_imgs_dir, 'desaparecido')
found_dir = os.path.join(pets_imgs_dir, 'procurase_dono')

lost_species_dir = os.listdir(lost_dir)
found_species_dir = os.listdir(found_dir)

lost_dirs = [os.path.join(lost_dir, species) for species in lost_species_dir]
found_dirs = [os.path.join(found_dir, species) for species in found_species_dir]

imgs_dirs = lost_dirs + found_dirs


interest_classes = ['dog', 'cat', 'horse', 'teddy bear']

for imgs_dir in imgs_dirs:
    img_files = os.listdir(imgs_dir)
    print(imgs_dir)
    num_vis_images = 30
    for img_file in tqdm(img_files):
        img_path = os.path.join(imgs_dir, img_file)
        yolo_sam(img_path, pets_imgs_dir)