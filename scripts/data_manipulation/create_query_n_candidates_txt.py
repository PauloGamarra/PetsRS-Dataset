import pandas as pd 
import sys


dataset_df = pd.read_csv(sys.argv[1])

count_samples = dict()

with open(sys.argv[2], 'w') as txt_file:
    for idx, row in dataset_df.iterrows():
        if not row['pet_id']  in count_samples.keys():
            count_samples[row['pet_id']] = 1
            query = row['img_path']
        else:
            count_samples[row['pet_id']] += 1

        if count_samples[row['pet_id']] == 2:
            candidate = row['img_path']
            txt_file.write('{} {}\n'.format(query, candidate))



