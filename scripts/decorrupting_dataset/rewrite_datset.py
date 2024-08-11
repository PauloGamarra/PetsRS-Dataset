import cv2
import os 
import sys

dataset_dir = sys.argv[1]

origins = os.listdir(dataset_dir)

count = 0

for origin in origins:
    origin_dir = os.path.join(dataset_dir, origin)
    species_dirs = os.listdir(origin_dir)
    for species in species_dirs:
        species_dir = os.path.join(origin_dir, species)
        img_files = os.listdir(species_dir)
        for img_file in img_files:
            count += 1
            print(count)
            img_path = os.path.join(species_dir, img_file)
            print(img_path)
            img = cv2.imread(img_path)
            if img is None:
                print('ERROR REWRITING {}'.format(img_path))
            else:
                cv2.imwrite(img_path, img)

