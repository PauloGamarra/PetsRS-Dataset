import os
import sys
import pandas as pd

from tqdm import tqdm  


pets_imgs_dir = sys.argv[1]

lost_dir = os.path.join(pets_imgs_dir, 'desaparecido')
found_dir = os.path.join(pets_imgs_dir, 'procurase_dono')

lost_species_dir = os.listdir(lost_dir)
found_species_dir = os.listdir(found_dir)

lost_dirs = [os.path.join(lost_dir, species) for species in lost_species_dir]
found_dirs = [os.path.join(found_dir, species) for species in found_species_dir]

imgs_df = pd.DataFrame(columns = ['img_path'])

imgs_dirs = lost_dirs + found_dirs 

for imgs_dir in imgs_dirs:
    img_files = os.listdir(imgs_dir)
    print(imgs_dir)
    for img_file in tqdm(img_files):
        img_path = os.path.join(imgs_dir, img_file)
        img_path = img_path.replace(pets_imgs_dir,'pets_rs_imgs')
        imgs_df = imgs_df.append({'img_path':img_path}, ignore_index=True)

imgs_df.to_csv('pets_dups_removed.csv', index=False)
