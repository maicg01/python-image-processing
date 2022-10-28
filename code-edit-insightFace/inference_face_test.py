from fileinput import filename
import imp
import onnxruntime
import torch
import torchvision
import cv2
from torchvision import transforms as T
import numpy as np


filename = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/onnx/ImageClassifier.onnx'
img = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/demo/rj1_test1/frame71_0.7_0.95.jpg'
img = cv2.imread(img)
def infer_face(path_filename, img):
    #chuyen ve kich thuoc phu hop
    img = cv2.resize(img, (32,32))

    input_size = tuple(img.shape[0:2][::-1])
    blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True) #chuyen ve dinh dang dung
    print(blob.shape)

    if torch.cuda.is_available():
        session = onnxruntime.InferenceSession(path_filename, None, providers=["CUDAExecutionProvider"])
    else:
        session = onnxruntime.InferenceSession(path_filename, None)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # print(input_name)
    # print(output_name)

    result = session.run([output_name], {input_name: blob})
    prediction=int(np.argmax(np.array(result).squeeze(), axis=0))
    print(prediction)
    return prediction

infer_face(filename, img)