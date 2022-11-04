import zipfile
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

def prepare_data():
    train_zip = zipfile.ZipFile('/home/maicg/Documents/python-image-processing/code-edit-insightFace/demo2/datatrain1.zip')
    file_names = [fname[-8:-4] for fname in train_zip.namelist() if '.jpg' in fname]
    print('Loading images metadata...')
    print(file_names)
    images = [Image.open(train_zip.open(fname)) for fname in train_zip.namelist() if '.jpg' in fname]
    print('Reading pixels...')
    for i, img in enumerate(images):
        img.load()
        if i % 100 == 0:
            print(i)
    [img.load() for img in images]
    print('Done', len(images))

    return images, file_names

images, file_names = prepare_data()

k=0
for score in file_names:
    img = np.array(images[file_names.index(score)])
    img = img[:,:,::-1]
    if '_' in score:
        score=score[1:]
        if float(score) < 0.65:
            cv2.imwrite('./demo2/filter/filter_data1/frame{0}_{1}.jpg'.format(k, round(float(score), 2)), img)
        else:
            cv2.imwrite('./demo2/filter/filter_dataOK/frame{0}_{1}.jpg'.format(k, round(float(score), 2)), img)

    elif float(score) < 0.65:
        # pass
        if float(score) == 0:
            print("er")
        else:
            cv2.imwrite('./demo2/filter/filter_data1/frame{0}_{1}.jpg'.format(k, round(float(score), 2)), img)
    else:
        cv2.imwrite('./demo2/filter/filter_dataOK/frame{0}_{1}.jpg'.format(k, round(float(score), 2)), img)
    
    k=k+1