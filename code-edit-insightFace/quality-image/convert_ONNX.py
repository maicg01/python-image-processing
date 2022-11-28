#convert to onnx runtime
import torch.onnx 
from torch.autograd import Variable
from models.model_resnet import ResNet, FaceQuality

import onnxruntime
import cv2
import numpy as np


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

def main():
    #convert to onnx
    # Let's load the model we just created and test the accuracy per label 
    model = ResNet(num_layers=100, feature_dim=512)
    path_resnet = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/quality-image/face_quality_model/backbone.pth'
    load_state_dict(model, torch.load(path_resnet))
    model.eval()
    print("doneeeee")
 
    # Conversion to ONNX 
    Convert_ONNX(model) 

    # convert Quality model to onnx
    model_quality = FaceQuality(512 * 7 * 7)
    path_quality = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/quality-image/face_quality_model/quality.pth'
    load_state_dict(model_quality, torch.load(path_quality))
    model_quality.eval()
    Convert_ONNX_1Output(model_quality)

    #==============================================================================================
    #chay inference tren onnx
    session = onnxruntime.InferenceSession('./onnx/Resnet2F.onnx', providers=['CUDAExecutionProvider'])
    first_input_name = session.get_inputs()[0].name
    first_output_name = session.get_outputs()[0].name
    print(first_input_name, first_output_name)

    input_cfg = session.get_inputs()[0]
    output_cfg = session.get_outputs()[0]
    print(input_cfg, output_cfg)

    #test data
    img = cv2.imread("/home/maicg/Documents/python-image-processing/code-edit-insightFace/quality-image/test_quality/quality_result_bad/frame108_0.0099.jpg")
    # img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        ccropped = img.swapaxes(1, 2).swapaxes(0, 1)
    except:
        print('error')
        return
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    # ccropped = torch.from_numpy(ccropped)
    # print(ccropped.shape)

    results = session.run(['output_0', 'output_1'], {'input': ccropped}) #input phai la mot array, kp torch
    print('results output', len(results))
    
    feature = results[0]
    quality = results[1]
    # print(feature, quality)
    print(len(feature[0]), len(quality[0]))

    #==============================================================================================
    # chay quality inference
    session1 = onnxruntime.InferenceSession('./onnx/Quality.onnx', providers=['CUDAExecutionProvider'])
    first_input_name1 = session1.get_inputs()[0].name
    first_output_name1 = session1.get_outputs()[0].name
    print(first_input_name1, first_output_name1)

    input_cfg1 = session1.get_inputs()[0]
    output_cfg1 = session1.get_outputs()[0]
    print(input_cfg1, output_cfg1)

    print("quality type: ", type(quality))
    print("quality shape: ", quality.shape)
    quality = np.array(quality, dtype = np.float32)
    results1 = session1.run(['output'], {'input': quality}) #input phai la mot array, kp torch
    
    rs_quality = results1[0]
    print(results1)

    # #test voi model chua convert onnx
    # quality_input = FaceQuality(512 * 7 * 7)
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # quality = torch.from_numpy(quality)
    # print('dau vao rs2', quality.shape)
    # rs2 = quality_input(quality)
    # print(rs2)


main()