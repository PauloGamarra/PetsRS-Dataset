from PIL import Image
import os
import sys
import ultralytics
from tqdm import tqdm


detection_model = ultralytics.YOLO("yolov10x.pt")

interest_classes = ['dog', 'cat']

def run_yolo(img_file):
    img = Image.open(img_file)
    results = detection_model.predict(img)
    result = results[0]
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        if class_id in interest_classes:
            return True, result.boxes
    return False, result.boxes
    
def save_no_pet(img_file, bboxes):
    with open('no_pet_imgs.txt', 'a') as file:
        bbox_classes = ' '.join([detection_model.names[box.cls[0].item()] for box in bboxes])
        no_pet_entry = '{} {}\n'.format(img_file, bbox_classes)
        file.write(no_pet_entry)
 
def save_yolo_detections(img_file, bboxes):
    with open('yolo_detections.txt', 'a') as file:
        bbox_classes = ' '.join([detection_model.names[box.cls[0].item()] for box in bboxes])
        entry = '{} {}\n'.format(img_file, bbox_classes)
        file.write(entry)

pets_imgs_dir = sys.argv[1]

lost_dir = os.path.join(pets_imgs_dir, 'desaparecido')
found_dir = os.path.join(pets_imgs_dir, 'procurase_dono')

lost_species_dir = os.listdir(lost_dir)
found_species_dir = os.listdir(found_dir)

lost_dirs = [os.path.join(lost_dir, species) for species in lost_species_dir]
found_dirs = [os.path.join(found_dir, species) for species in found_species_dir]

imgs_dirs = lost_dirs + found_dirs

for imgs_dir in imgs_dirs:
    img_files = os.listdir(imgs_dir)
    print(imgs_dir)
    for img_file in tqdm(img_files):
        img_path = os.path.join(imgs_dir, img_file)
        has_pet, bboxes = run_yolo(img_path)
        if not has_pet:
            save_no_pet(img_path, bboxes)
        save_yolo_detections(img_path, bboxes)
