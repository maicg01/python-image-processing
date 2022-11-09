from facenet_pytorch import MTCNN, InceptionResnetV1
import torch 
from torchvision import transforms
import math
import cv2
import time
import os

# time_start = time.time()
# # If required, create a face detection pipeline using MTCNN:
# mtcnn = MTCNN(image_size=160, margin=0)

# # Create an inception resnet (in eval mode):
# resnet = InceptionResnetV1(pretrained='vggface2').eval()

# from PIL import Image

# img = Image.open('/home/maicg/Documents/python-image-processing/angelina.jpg')

# # Get cropped and prewhitened image tensor
# img_cropped = mtcnn(img, save_path='face.png')
# print(img_cropped.size())

# # Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(img_cropped.unsqueeze(0))

# # print(img_embedding)
# print(img_embedding.size())

# # # Or, if using for VGGFace2 classification
# # resnet.classify = True
# # img_probs = resnet(img_cropped.unsqueeze(0))
# # print(img_probs)

# img2 = Image.open('/home/maicg/Documents/python-image-processing/img2.jpg')

# # Get cropped and prewhitened image tensor
# img_cropped2 = mtcnn(img2, save_path='face2.png')

# # Calculate embedding (unsqueeze to add batch dimension)
# img_embedding2 = resnet(img_cropped2.unsqueeze(0))

# # print(img_embedding2)
# print(img_embedding2.size())

# cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
# output = cos(img_embedding, img_embedding2)
# print("goc ti le giua anh 1 va 2: ", output)
# time_end = time.time()
# avr_time = time_end-time_start
# print("=================", avr_time)

# # ========================================================================
# img3 = Image.open('/home/maicg/Documents/python-image-processing/img3.jpg')

# # Get cropped and prewhitened image tensor
# img_cropped3 = mtcnn(img3, save_path='face3.png')
# # print(type(img_cropped3))

# # Calculate embedding (unsqueeze to add batch dimension)
# img_embedding3 = resnet(img_cropped3.unsqueeze(0))

# # print(img_embedding2)
# print(img_embedding3.size())

# cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
# output2 = cos(img_embedding, img_embedding3)
# print("goc ti le giua anh 1 va 3: ", output2)

# # ========================================================================
# img4 = Image.open('/home/maicg/Documents/python-image-processing/img4.jpg')

# # Get cropped and prewhitened image tensor
# img_cropped4 = mtcnn(img4, save_path='face4.png')

# # Calculate embedding (unsqueeze to add batch dimension)
# img_embedding4 = resnet(img_cropped4.unsqueeze(0))

# # print(img_embedding2)
# print(img_embedding4.size())

# cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
# output3 = cos(img_embedding, img_embedding4)
# print("goc ti le giua anh 1 va 4: ", output3)
# print(math.acos(output))
# print(math.acos(output2))
# print(math.acos(output3))



# img1 = Image.open('/home/maicg/Documents/python-image-processing/code-edit-insightFace/facenet_al/data_test/img1.jpg')
# img2 = Image.open('/home/maicg/Documents/python-image-processing/code-edit-insightFace/facenet_al/data_test/img3.jpg')

# def fixed_image_standardization(image_tensor):
#     processed_tensor = (image_tensor - 127.5) / 128.0
#     return processed_tensor

# #viet ham facenet dungcosin similary
# def computeCosin(img1, img2):
#     convert_tensor = transforms.ToTensor()
#     # convert_tensor = transforms.Compose([
#     #     transforms.ToTensor()
#     # ])
#     img1= img1.resize((160,160))
#     img2= img2.resize((160,160))
#     img1=convert_tensor(img1)
#     img2=convert_tensor(img2)
#     # img1=fixed_image_standardization(img1)
#     # img2=fixed_image_standardization(img2)
#     # Calculate embedding (unsqueeze to add batch dimension)
#     img_embedding = resnet(img1.unsqueeze(0))
#     # print(img_embedding)
#     print(img_embedding.size())

#     # Calculate embedding (unsqueeze to add batch dimension)
#     img_embedding2 = resnet(img2.unsqueeze(0))
#     # print(img_embedding2)
#     print(img_embedding2.size())

#     cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#     output = cos(img_embedding, img_embedding2)
#     # print("goc ti le giua anh 1 va 2: ", output)
#     return output

# # result = computeCosin(img1, img2)
# # print("KQ TINH BOI HAM: ", result)


# img3 = Image.open('/home/maicg/Documents/python-image-processing/code-edit-insightFace/facenet_al/face.png')
# img4 = Image.open('/home/maicg/Documents/python-image-processing/code-edit-insightFace/facenet_al/face2.png')
# result2 = computeCosin(img3,img4)
# print("KQ TINH BOI HAM img3&4: ", result2)
# #ket luan: ko dung MTCNN thi giam do chinh xac


def computeCosin(img1, img2):
    mtcnn = MTCNN(image_size=160, margin=0)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    from PIL import Image

    # Get cropped and prewhitened image tensor
    img_cropped1 = mtcnn(img1)
    # print(img_cropped.size())

    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding1 = resnet(img_cropped1.unsqueeze(0))

    #img2
    # Get cropped and prewhitened image tensor
    img_cropped2 = mtcnn(img2)
    # print(img_cropped.size())

    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding2 = resnet(img_cropped2.unsqueeze(0))

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(img_embedding1, img_embedding2)
    # print("goc ti le giua anh 1 va 2: ", output)
    return output


def compare_MTCNN(img1, img2):
    results = computeCosin(img1, img2)
    # print("==========KQ run: ", results)
    return results

def main():
    # img1 = ['/home/maicg/Documents/python-image-processing/angelina.jpg']
    # img2 = ['/home/maicg/Documents/python-image-processing/img2.jpg']
    # img3 = ['/home/maicg/Documents/python-image-processing/img3.jpg']
    # img4 = ['/home/maicg/Documents/python-image-processing/img4.jpg']

    # compare_image(img1, img2)
    # compare_image(img1, img3)
    # compare_image(img1, img4)
    path = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/demo2/test-facenet'
    img1 = cv2.imread('/home/maicg/Documents/python-image-processing/sample.jpg')
    count = 0
    avr_time = 0

    time_start = time.time()
    for img in os.listdir(path):
        pathName = os.path.join(path,img)
        # print('done')
        img2 = cv2.imread(pathName)
        result = compare_MTCNN(img1, img2)
        print(result)
        count = count + 1

    time_end = time.time()
    avr_time = round(((time_end-time_start)/count), 2)
    print(avr_time)

main()

