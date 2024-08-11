import os 
import sys

duplicates_txt_pth = sys.argv[1]

def get_pet_id(img_path):
    img_name = os.path.basename(img_path).replace('.png','')
    pet_id = img_name.split('_')[0]
    return pet_id

def get_pet_imgs(img_path):
    pet_id = get_pet_id(img_path)
    imgs_dir = os.path.dirname(img_path)
    pet_imgs = [img for img in os.listdir(imgs_dir) if get_pet_id(img) == pet_id]
    print('images of {}:'.format(pet_id))
    print(pet_imgs)
    return pet_imgs

def get_pet_num_imgs(img_path):
    return len(get_pet_imgs(img_path))

def get_pet_with_more_imgs(duplicate):
    pet_imgs = duplicate.copy()
    pet_imgs.sort(key=lambda x: get_pet_num_imgs(x))

    print('{} has more images'.format(pet_imgs[-1]))

    return pet_imgs[-1]

with open(duplicates_txt_pth, 'r') as duplicates_txt:
    duplicates = duplicates_txt.readlines()
    duplicates = [duplicate.strip().split() for duplicate in duplicates]

for duplicate in duplicates:
    print('=========================================================')
    print(duplicate)
    duplicate_copy = duplicate.copy()
    for img_path in duplicate_copy:
        if not os.path.exists(img_path):
            print('file already deleted: {}\nremoving from duplicate'.format(img_path))
            duplicate.remove(img_path)
    if len(duplicate) <= 1:
        print('only one file in duplicate. skipping...')
        continue
    pet_ids = [get_pet_id(img_path) for img_path in duplicate]
    print(pet_ids)
    if len(set(pet_ids)) == 1:
        print('all images belong to the same pet')
        ref_img = duplicate[0]
        print('using {} as ref image'.format(ref_img))
        for img_path in duplicate[1:]:
            print('removing {}'.format(img_path))
            os.remove(img_path)
        continue
    ref_img = get_pet_with_more_imgs(duplicate)
    duplicate.remove(ref_img)
    for img_path in duplicate:
        print('removing {} imgs'.format(img_path))
        for pet_img_path in get_pet_imgs(img_path):
            imgs_dir = os.path.dirname(img_path)
            pet_img_path = os.path.join(imgs_dir, pet_img_path)
            print('removing {}'.format(pet_img_path))
            os.remove(pet_img_path)

    

