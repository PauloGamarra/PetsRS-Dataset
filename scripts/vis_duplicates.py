import sys
import cv2

duplicates_txt = sys.argv[1]

with open(duplicates_txt, 'r') as dup_file:
    lines = dup_file.readlines()
    for duplicates in lines:
        duplicates = duplicates.split()
        for idx, duplicate in enumerate(duplicates):
            print(idx, duplicate)
            img = cv2.imread(duplicate)
            cv2.imshow('{}'.format(idx), img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

cv2.destroyAllWindows()

