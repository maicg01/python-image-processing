import torch
import cv2
from models.model_resnet import ResNet, FaceQuality
import os
import argparse
import shutil
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

def get_face_quality(backbone, quality, device, img):
    resized = cv2.resize(img, (112, 112))
    ccropped = resized[...,::-1] # BGR to RGB
    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    # extract features
    backbone.eval() # set to evaluation mode
    with torch.no_grad():
        x, fc = backbone(ccropped.to(device), True) # x la dau ra chuan hoa cua Resnet, fc la vecto danh gia chat luong khung hinh
        s = quality(fc)[0]

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
        checkpoint = torch.load('./face_quality_model/backbone.pth', map_location='cpu')
        load_state_dict(BACKBONE, checkpoint)
    else:
        print("No Checkpoint Found at '{}' Please Have a Check or Continue to Train from Scratch".format('./face_quality_model/backbone.pth'))
        return
    if os.path.isfile('./face_quality_model/quality.pth'):
        print("Loading Quality Checkpoint '{}'".format('./face_quality_model/quality.pth'))
        checkpoint = torch.load('./face_quality_model/quality.pth', map_location='cpu')
        load_state_dict(QUALITY, checkpoint)
    else:
        print("No Checkpoint Found at '{}' Please Have a Check or Continue to Train from Scratch".format(args.quality))
        return
    BACKBONE.to(DEVICE)
    QUALITY.to(DEVICE)
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
def take_image(BACKBONE, QUALITY, DEVICE, output_path, image):
    # output_path = 'quality_result'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    # file_test_path = 'test_me/frameNEW126_0.6_0.62_0.81.jpg'
    quality, fc_out = get_face_quality(BACKBONE, QUALITY, DEVICE, image)
    cv2.imwrite('{}/{:.4f}.jpg'.format(output_path, quality[0]), image)
    return quality, fc_out



def main():
    BACKBONE, QUALITY, DEVICE = load_net()

    output_path = 'quality_result'
    file_test_path = 'test_me'

    quality, emb = take_output(BACKBONE, QUALITY, DEVICE, output_path=output_path, file_test_path=file_test_path)
    print(quality, emb)

main()


