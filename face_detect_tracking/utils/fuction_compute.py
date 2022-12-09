import torch
import cv2
from models.model_resnet import ResNet, FaceQuality
import os
import argparse
import shutil
import numpy as np
import torch.onnx 
from torch.autograd import Variable
import onnxruntime

def load_state_dict(model, state_dict):
    all_keys = {k for k in state_dict.keys()}
    for k in all_keys:
        if k.startswith('module.'):
            state_dict[k[7:]] = state_dict.pop(k)
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    if len(pretrained_dict) == len(model_dict):
        print("all params loaded")
    else:
        not_loaded_keys = {k for k in pretrained_dict.keys() if k not in model_dict.keys()}
        print("not loaded keys:", not_loaded_keys)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def get_face_quality(backbone, quality, device, img):
    resized = cv2.resize(img, (112, 112))
    # ccropped = resized[...,::-1] # BGR to RGB
    # load numpy to tensor
    try:
        ccropped = img.swapaxes(1, 2).swapaxes(0, 1)
    except:
        print('erorr')
        return
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    # extract features
    backbone.eval() # set to evaluation mode
    with torch.no_grad():
        x, fc = backbone(ccropped.to(device), True) # x la dau ra chuan hoa cua Resnet, fc la vecto danh gia chat luong khung hinh
        s = quality(fc)[0]
        # print('================', s)
        # print('================', fc.shape)

    return s.cpu().numpy(), x #x la vecto dac trung 512 chieu, s la vecto dung de tinh quality


def load_net():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BACKBONE = ResNet(num_layers=100, feature_dim=512)
    # print('backbone', BACKBONE)
    print("loai cua backbone: ", type(BACKBONE))
    QUALITY = FaceQuality(512 * 7 * 7)
    print("loai cua QUALITY: ", type(QUALITY))


    if os.path.isfile('./face_quality_model/backbone.pth'):
        print("Loading Backbone Checkpoint '{}'".format('./face_quality_model/backbone.pth'))
        checkpoint = torch.load('./face_quality_model/backbone.pth')
        load_state_dict(BACKBONE, checkpoint)
    else:
        print("No Checkpoint Found at '{}' Please Have a Check or Continue to Train from Scratch".format('./face_quality_model/backbone.pth'))
        return
    if os.path.isfile('./face_quality_model/quality.pth'):
        print("Loading Quality Checkpoint '{}'".format('./face_quality_model/quality.pth'))
        checkpoint = torch.load('./face_quality_model/quality.pth')
        load_state_dict(QUALITY, checkpoint)
    else:
        print("No Checkpoint Found at '{}' Please Have a Check or Continue to Train from Scratch".format(args.quality))
        return
    BACKBONE.to(DEVICE)
    QUALITY.to(DEVICE)
    print('device ', DEVICE)
    BACKBONE.eval()
    QUALITY.eval()
    print("done")
    return BACKBONE, QUALITY, DEVICE

#take output cua folder hoac cua anh voi doi so file_test_path la duong dan cua folder hoac duong dan cua anh
def take_output(BACKBONE, QUALITY, DEVICE, output_path, file_test_path):
    # output_path = 'quality_result'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # file_test_path = 'test_me/frameNEW126_0.6_0.62_0.81.jpg'

    if os.path.isfile(file_test_path):
        image = cv2.imread(file_test_path)
        if image is None or image.shape[0] == 0:
            print("Open image failed: ", file_test_path)
            return
        quality, fc_out = get_face_quality(BACKBONE, QUALITY, DEVICE, image)
        cv2.imwrite('{}/{:.4f}.jpg'.format(output_path, quality[0]), image)
        return quality, fc_out
    elif os.path.isdir(file_test_path):
        for tmp in os.listdir(file_test_path):
            image = cv2.imread(os.path.join(file_test_path, tmp))
            if image is None or image.shape[0] == 0:
                print("Open image failed: ", file_test_path)
                continue
            quality, fc_out = get_face_quality(BACKBONE, QUALITY, DEVICE, image)
            # print(quality, fc_out.shape)
            cv2.imwrite('{}/{:.4f}.jpg'.format(output_path, quality[0]), image)
            return quality, fc_out
    else:
        print(file_test_path, "not exists")
        return

#take output image da dung cv2.imread()
def take_image(BACKBONE, QUALITY, DEVICE, image):
    # output_path = 'quality_result'
    # if os.path.exists(output_path):
    #     shutil.rmtree(output_path)
    # os.makedirs(output_path)

    # file_test_path = 'test_me/frameNEW126_0.6_0.62_0.81.jpg'
    try:
        quality, fc_out = get_face_quality(BACKBONE, QUALITY, DEVICE, image)
    except:
        print("error take image")
        return
    # cv2.imwrite('{}/{:.4f}.jpg'.format(output_path, quality[0]), image)
    return quality, fc_out

def computeCosinQuality(emb1, emb2):
    # emb2 = computeEmb(img2)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(emb1, emb2)
    # print("goc ti le giua anh 1 va 2: ", output)
    return output



#viet ham chay onnx
#Function to Convert to ONNX with 2 output
def Convert_ONNX(model): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = Variable(torch.randn(1, 3, 112, 112))

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "./onnx/Resnet2F.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output_0', 'output_1'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

#Function to Convert to ONNX with 1 output
def Convert_ONNX_1Output(model): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = Variable(torch.randn(1, 25088))

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "./onnx/Quality.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

def load_model_onnx(file_name):
    session = onnxruntime.InferenceSession(file_name, providers=['CUDAExecutionProvider'])
    return session

def process_onnx(img, session1, session2):
    # session1 = load_model_onnx('./onnx/Resnet2F.onnx')
    # session2 = load_model_onnx('./onnx/Quality.onnx')
    # session1 model resnet co 2 output gom feature va vecto quality
    # session2 la mmodel quality co output la chat luong cua buc anh

    # chuan hoa dau vao
    img = cv2.resize(img, (112, 112))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        ccropped = img.swapaxes(1, 2).swapaxes(0, 1)
    except:
        print('error')
        return
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0

    results1 = session1.run(['output_0', 'output_1'], {'input': ccropped}) #input phai la mot array, kp torch
    print('results output', len(results1))
    
    feature = results1[0]
    quality = results1[1]

    results2 = session2.run(['output'], {'input': quality})

    rs_quality = results2[0]

    feature = torch.from_numpy(feature)
    print("shape cua feature: ", type(feature))

    return rs_quality[0], feature



# def main():
#     BACKBONE, QUALITY, DEVICE = load_net()

#     output_path = 'quality_result'
#     file_test_path = 'test_me'

#     quality, emb = take_output(BACKBONE, QUALITY, DEVICE, output_path=output_path, file_test_path=file_test_path)
#     print(quality, emb)

# main()


