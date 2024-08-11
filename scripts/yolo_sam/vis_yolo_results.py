import sys
import cv2 
import os

results_path = sys.argv[1]

with open(results_path, 'r') as results_file:
    lines = results_file.readlines()

for idx, result in enumerate(lines):
    img_path = result.split()[0].replace('/data/','')
    img_path = os.path.join('/home/paulogamarra/data/tcc/pets_dataset_imgs',img_path)
    classes = result.split()[1:]
    print(idx, img_path, classes)
    
    img = cv2.imread(img_path)
    cv2.imshow('img', img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
