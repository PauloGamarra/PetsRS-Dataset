import os 
import shutil
import sys

data_dir = sys.argv[2]
dst_dir = sys.argv[3]


if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

with open(sys.argv[1]) as txt_file:
    pairs = [tuple(pair.strip().split(' ')) for pair in txt_file.readlines()]

for pair in pairs:
    query_path = pair[0].replace('pets_rs_imgs/', '')
    candidate_path = pair[1].replace('pets_rs_imgs/', '')
    
    src_query_path = os.path.join(data_dir,query_path)
    dst_query_dir = os.path.join(dst_dir, os.path.dirname(query_path))
    dst_query_dir = os.path.join(dst_query_dir, 'query')

    src_candidate_path = os.path.join(data_dir, candidate_path)
    dst_candidate_dir = os.path.join(dst_dir, os.path.dirname(candidate_path))
    dst_candidate_dir = os.path.join(dst_candidate_dir, 'candidate')
    
    pet_id = os.path.basename(query_path).split('.')[0].split('_')[0]

    dst_query_path = os.path.join(dst_query_dir, pet_id + '.png')
    dst_candidate_path = os.path.join(dst_candidate_dir, pet_id + '.png')

    if not os.path.exists(dst_query_dir):
        os.makedirs(dst_query_dir)
    if not os.path.exists(dst_candidate_dir):
        os.makedirs(dst_candidate_dir)

    shutil.copy(src_query_path, dst_query_path)
    shutil.copy(src_candidate_path, dst_candidate_path)
