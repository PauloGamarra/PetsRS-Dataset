import pandas as pd
import sys 

pets_df = pd.read_csv(sys.argv[1])

expanded_df = pets_df['img_path'].str.split('/', expand=True)

pets_df['species'] = expanded_df[2]
pets_df['origin'] = expanded_df[1]

expanded_fname = expanded_df[3].str.split('_', expand=True)
expanded_fname.replace('.png', '')
pet_id = expanded_fname[0].str.replace('.png', '')
img_id = expanded_fname[1].str.replace('.png', '')
img_id = img_id.fillna(value='1')

pets_df['pet_id'] = pd.to_numeric(pet_id)
pets_df['img_id'] = pd.to_numeric(img_id)

pets_df = pets_df.sort_values(by=['origin', 'species', 'pet_id', 'img_id'])

pets_df['single_image'] = ~pets_df.duplicated(keep=False, subset='pet_id')

dataset_pets = pets_df[~pets_df['single_image']].reset_index()
dataset_pets = dataset_pets.drop(columns='single_image')
dataset_pets = dataset_pets.drop(columns=['index'], axis=1)

counts = dataset_pets['pet_id'].value_counts()
counts = counts.to_dict()
dataset_pets['num_images'] = dataset_pets['pet_id'].map(counts) 

dataset_pets.to_csv(sys.argv[2], index=False)
