import sys
import pandas as pd 

csv_path = sys.argv[1]

pets_df = pd.read_csv(csv_path)
unique_pets_df = pets_df.drop_duplicates(subset=['pet_id'])


print('total:')
print('total: {} / {}'.format(len(unique_pets_df), len(pets_df)))
counts = pets_df['species'].value_counts()
unique_counts = unique_pets_df['species'].value_counts()
for species in counts.keys():
    print('{}: {} / {}'.format(species, unique_counts[species], counts[species]))


for origin in ['procurase_dono', 'desaparecido']:
        subset_df = pets_df[pets_df['origin'] == origin]
        unique_pets_df = subset_df.drop_duplicates(subset=['pet_id'])


        print('{}:'.format(origin))
        print('total: {} / {}'.format(len(unique_pets_df), len(subset_df)))
        counts = subset_df['species'].value_counts()
        unique_counts = unique_pets_df['species'].value_counts()
        for species in ['cao', 'gato', 'cavalo', 'passaro', 'outro']:
            if not species in unique_counts.keys():
                unique_counts[species] = 0
            if not species in counts.keys():
                counts[species] = 0
            print('{}: {} / {}'.format(species, unique_counts[species], counts[species]))

mean_images = pets_df.drop_duplicates(subset=['pet_id'])['num_images'].mean()
median_images = pets_df.drop_duplicates(subset=['pet_id'])['num_images'].median()

print('num_images stats (mean / median)')
print('total: {} / {}'.format(mean_images, median_images))

for species in ['cao', 'gato', 'cavalo', 'passaro', 'outro']:
    species_df = pets_df[pets_df['species'] == species]
    mean_images = species_df.drop_duplicates(subset=['pet_id'])['num_images'].mean()
    median_images = species_df.drop_duplicates(subset=['pet_id'])['num_images'].median()
    print('{}: {} / {}'.format(species, mean_images, median_images))
