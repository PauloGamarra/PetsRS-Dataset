import shutil
import os
import sys
import pandas as pd

dataset_df = sys.argv[1]
original_dir = sys.argv[2]
dest_dir = sys.argv[3]

pets_df = pd.read_csv(dataset_df)
pets_df = pets_df[pets_df['species'].isin(['cao', 'gato'])]

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

for origin in ['desaparecido', 'procurase_dono']:
    origin_dir = os.path.join(dest_dir, origin)
    if not os.path.exists(origin_dir):
        os.mkdir(origin_dir)
    for species in ['cao', 'gato']:
        species_dir = os.path.join(origin_dir, species)
        if not os.path.exists(species_dir):
            os.mkdir(species_dir)

for idx, row in pets_df.iterrows():
    src_file = row['img_path']
    src_file = src_file.replace('pets_rs_imgs/','')
    src_path = os.path.join(original_dir, src_file)
    dst_path = os.path.join(dest_dir, src_file)
    print(src_path, dst_path)
    shutil.copy(src_path, dst_path)

