import os
import numpy as np
from PIL import Image
import face_recognition
import faiss
import time

ROOTDATA = 'dataset'

def read_file(path):
    img_path = []
    for paths, subdirs, files in os.walk(path):
        for name in files:
            img_path.append(os.path.join(paths,name))
    return img_path

train_imgs = read_file(ROOTDATA)
print(train_imgs)

faceEncode = []
img_paths = []
for path in train_imgs:
    # read img
    img = face_recognition.load_image_file(path)
    # detect face
    img_location = face_recognition.face_locations(img)
    if len(img_location) > 0:
        #crop face
        for (top,right,bottom,left) in img_location:
            face_img = img[top:bottom,left:right]
            # save face img
            pil_img = Image.fromarray(face_img)
            pil_img.save(path)
            # encode
            face_encode = face_recognition.face_encodings(img)[0]
            faceEncode.append(face_encode)
            img_paths.append(path)
print(len(faceEncode))

train_labels = np.array([img.split('/')[-2] for img in img_paths])
faceEncode = np.array(faceEncode,dtype=np.float32)
print(faceEncode.shape)

time_s = time.time()
# create index with faiss
face_index = faiss.IndexFlatL2(128)
# add vector
face_index.add(faceEncode)

#test
test_img = face_recognition.load_image_file('test/Hoang_anh(1).jpg')
img_location = face_recognition.face_locations(test_img)
if len(img_location) > 0:
    #crop face
    for (top,right,bottom,left) in img_location:
        face_img = test_img[top:bottom,left:right]
        qr = face_recognition.face_encodings(face_img)[0]
    
#convert to (n,128)
qr = np.array(qr,dtype=np.float32).reshape(-1,128)
w,result = face_index.search(qr, k=10)
label = [train_labels[i] for i in result[0]]
print(w[0][0])
print(label)
time_end = time.time()
t_a = time_end-time_s
print(t_a)