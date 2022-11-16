from facenet_pytorch import MTCNN
import torch 
from torchvision import transforms
import math
import cv2
import time
import os
from PIL import Image
from torch.utils import mkldnn as mkldnn_utils
from inceptionResnetV1 import InceptionResnetV1


def computeCosin(img1, img2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
    )

    net = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    x_test1, prob = mtcnn(img1, return_prob=True)
    # print(prob)
    print(x_test1.shape)

    x_aligned1=[]
    x_aligned1.append(x_test1)
    test_aligned1 = torch.stack(x_aligned1).to(device)
    test_embeddings1 = net(test_aligned1).detach().cpu()

    #img2
    x_test2, prob = mtcnn(img2, return_prob=True)
    # print(prob)
    # print(x_test.shape)

    x_aligned2=[]
    x_aligned2.append(x_test2)
    test_aligned2 = torch.stack(x_aligned2).to(device)
    test_embeddings2 = net(test_aligned2).detach().cpu()

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(test_embeddings1, test_embeddings2)
    print("goc ti le giua anh 1 va 2: ", output)
    return output


def compare_MTCNN(img1, img2):
    results = computeCosin(img1, img2)
    # print("==========KQ run: ", results)
    return results

def main():
    img1 = Image.open('/home/maicg/Documents/python-image-processing/angelina.jpg')
    img2 = Image.open('/home/maicg/Documents/python-image-processing/img2.jpg')
    img3 = Image.open('/home/maicg/Documents/python-image-processing/img3.jpg')
    img4 = Image.open('/home/maicg/Documents/python-image-processing/img4.jpg')

    # img1 = cv2.imread('/home/maicg/Documents/python-image-processing/angelina.jpg')
    # img2 = cv2.imread('/home/maicg/Documents/python-image-processing/img2.jpg')
    # img3 = cv2.imread('/home/maicg/Documents/python-image-processing/img3.jpg')
    # img4 = cv2.imread('/home/maicg/Documents/python-image-processing/img4.jpg')

    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    # img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

    computeCosin(img1, img2)
    computeCosin(img1, img3)
    computeCosin(img1, img4)
    

    # path = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/demo2/test-facenet'
    # img1 = cv2.imread('/home/maicg/Documents/python-image-processing/sample.jpg')
    # count = 0
    # avr_time = 0

    # time_start = time.time()
    # for img in os.listdir(path):
    #     pathName = os.path.join(path,img)
    #     # print('done')
    #     img2 = cv2.imread(pathName)
    #     result = compare_MTCNN(img1, img2)
    #     print(result)
    #     count = count + 1

    # time_end = time.time()
    # avr_time = round(((time_end-time_start)/count), 2)
    # print(avr_time)

main()

