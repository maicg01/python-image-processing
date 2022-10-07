import imp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
# from torch import models
from torch.autograd import Variable
from models import Net

os.chdir('/home/devai01/Me/python-image-processing/')
image = cv2.imread('people.jpg')


gray = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('file-haarcascade/haarcascade_frontalface_default.xml')

# load in a haar cascade classifier for detecting eyes
eye_cascade = cv2.CascadeClassifier('file-haarcascade/haarcascade_frontalface_default.xml')


gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

net = Net()

## load the best saved model parameters (by your path name)
## You'll need to un-comment the line below and add the correct name for *your* saved model
net.load_state_dict(torch.load('Face-keypoints/facial_keypoints_model.pt'))

## print out your net and prepare it for testing (uncomment the line below)
net.eval()

def show_all_keypoints(image, keypoints):  
    batch_size = len(image)
    for i, face in enumerate(image):
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the predicted key_pts data
        predicted_keypoints = keypoints[i].data
        predicted_keypoints = predicted_keypoints.numpy()
        # undo normalization of keypoints  
        predicted_keypoints = predicted_keypoints*50.0+100

        # os.chdir('/home/devai01/Me/python-image-processing/results')
        # cv2.imwrite("face-detection-keypoints.jpg", face)

        plt.imshow(face, cmap='gray')
        plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], s=20, marker='.', c='m')
        
        plt.axis('off')

    plt.show()

image_copy = np.copy(image)
#Including a padding to extract face as  HAAR classifier's bounding box, crops sections of the face

PADDING = 40
images, keypoints = [], []

# loop over the detected faces from your haar cascade
for (x,y,w,h) in faces:
    
    # Select the region of interest that is the face in the image 
    roi = image_copy[y-PADDING:y+h+PADDING, x-PADDING:x+w+PADDING]
    
    ## Convert the face region from RGB to grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    ## Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi = (roi / 255.).astype(np.float32)
    
    ## Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    roi = cv2.resize(roi, (224, 224))
    images.append(roi)
    
    ## Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    if len(roi.shape) == 2:
        roi = np.expand_dims(roi, axis=0)
    else:
        roi = np.rollaxis(roi, 2, 0)
        
    # Match the convolution dimensions
    roi = np.expand_dims(roi, axis=0)
    
    ## Make facial keypoint predictions using your loaded, trained network 
    # Forward pass
    roi = torch.from_numpy(roi).type(torch.FloatTensor)
    output_pts = net.forward(roi)
    
    output_pts = output_pts.view(output_pts.size()[0], 68, -1)
    keypoints.append(output_pts[0])
    
## Display each detected face and the corresponding keypoints
show_all_keypoints(images, keypoints)





# os.chdir('/home/devai01/Me/python-image-processing/results')
# cv2.imwrite("face-detection-MTCNN.jpg", image)