from facenet_pytorch import MTCNN, InceptionResnetV1
import torch 
import math

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=160, margin=0)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

from PIL import Image

img = Image.open('/home/maicg/Documents/python-image-processing/angelina.jpg')

# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img, save_path='face.png')

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))

# print(img_embedding)
print(img_embedding.size())

# # Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))
# print(img_probs)

img2 = Image.open('/home/maicg/Documents/python-image-processing/img2.jpg')

# Get cropped and prewhitened image tensor
img_cropped2 = mtcnn(img2, save_path='face2.png')

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding2 = resnet(img_cropped2.unsqueeze(0))

# print(img_embedding2)
print(img_embedding2.size())

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(img_embedding, img_embedding2)
print("goc ti le giua anh 1 va 2: ", output)

# ========================================================================
img3 = Image.open('/home/maicg/Documents/python-image-processing/img3.jpg')

# Get cropped and prewhitened image tensor
img_cropped3 = mtcnn(img3, save_path='face3.png')

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding3 = resnet(img_cropped3.unsqueeze(0))

# print(img_embedding2)
print(img_embedding3.size())

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
output2 = cos(img_embedding, img_embedding3)
print("goc ti le giua anh 1 va 3: ", output2)

# ========================================================================
img4 = Image.open('/home/maicg/Documents/python-image-processing/img4.jpg')

# Get cropped and prewhitened image tensor
img_cropped4 = mtcnn(img4, save_path='face4.png')

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding4 = resnet(img_cropped4.unsqueeze(0))

# print(img_embedding2)
print(img_embedding4.size())

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
output3 = cos(img_embedding, img_embedding4)
print("goc ti le giua anh 1 va 4: ", output3)
print(math.acos(output))
print(math.acos(output2))
print(math.acos(output3))