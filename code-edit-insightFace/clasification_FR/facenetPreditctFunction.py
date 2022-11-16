#thu vien can thiet
from __future__ import division
from calendar import c
import datetime
from operator import lt
import numpy as np
import onnx
import onnxruntime
import os
import os.path as osp
import cv2
import sys
import math
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
from PIL import Image
import time
from numpy import asarray
from numpy import expand_dims



# def computeCosin(emb1, img2):
#     emb2 = computeEmb(img2)
#     cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#     output = cos(emb1, emb2)
#     # print("goc ti le giua anh 1 va 2: ", output)
#     return output

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def alignment(img, l_eye, r_eye):
    left_eye_x = l_eye[0]; left_eye_y = l_eye[1]
    right_eye_x = r_eye[0]; right_eye_y = r_eye[1]

    # cv2.circle(img, left_eye_center, 2, (255, 0, 0) , 2)
    # cv2.circle(img, right_eye_center, 2, (255, 0, 0) , 2)
    # cv2.line(img,right_eye_center, left_eye_center,(67,67,67),2)
    # plt.imshow(img[:,:,::-1])
    # plt.show()

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
        print("rotate to clock direction")
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
        # print("rotate to inverse clock direction")

    a = euclidean_distance(l_eye, point_3rd)
    b = euclidean_distance(r_eye, l_eye)
    c = euclidean_distance(r_eye, point_3rd)

    cos_a = (b*b + c*c - a*a)/(2*b*c)
    # print("cos(a) = ", cos_a)
    
    angle = np.arccos(cos_a)
    # print("angle: ", angle," in radian")
    
    angle = (angle * 180) / math.pi
    # print("angle: ", angle," in degree")

    if direction == -1:
        angle = 90 - angle

    from PIL import Image
    new_img = Image.fromarray(img)
    new_img = np.array(new_img.rotate(direction * angle))
    
    return new_img

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class SCRFD:
    def __init__(self, model_file=None, session=None):
        import onnxruntime
        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        self.batched = False
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(self.model_file, providers=['CUDAExecutionProvider'])
        self.center_cache = {}
        self.nms_thresh = 0.4
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs)==6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs)==9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs)==10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs)==15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        # if ctx_id<0:
        #     self.session.set_providers(['CPUExecutionProvider'])
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in scrfd model, ignore')
            else:
                self.input_size = input_size

    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name : blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                #print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores>=thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                #kpss = kps_preds
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, thresh=0.5, input_size = None, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
            
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

def get_scrfd(name, download=False, root='~/.insightface/models', **kwargs):
    if not download:
        assert os.path.exists(name)
        return SCRFD(name)
    else:
        from .model_store import get_model_file
        _file = get_model_file("scrfd_%s" % name, root=root)
        return SCRFD(_file)


def scrfd_2p5gkps(**kwargs):
    return get_scrfd("2p5gkps", download=True, **kwargs)

def xyz_coordinates(kps):
    l_eye = kps[0]
    r_eye = kps[1]
    nose = kps[2]
    l_mouth = kps[3]
    r_mouth = kps[4]
    # kp = l4.astype(np.int)
    # cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
    center1 = (l_eye + l_mouth)/2
    # print('=======================center1',center1)
    center2 = (r_eye + r_mouth)/2
    distance12 = math.dist(center1,center2)
    # print('=======================distance12',distance12)
    
    distance_nose1 = math.dist(center1, nose)
    distance_nose2 = math.dist(center2, nose)

    center_eye = (l_eye + r_eye)/2
    center_mouth = (l_mouth + r_mouth)/2
    distance_center_eye_mouth =  math.dist(center_eye,center_mouth)
    distance_nose_ceye = math.dist(center_eye, nose)
    distance_nose_cmouth = math.dist(center_mouth, nose)

    distance_eye = math.dist(l_eye,r_eye)
    distance_mouth = math.dist(l_eye,r_eye)

    return distance12, distance_nose1, distance_nose2, distance_center_eye_mouth, distance_nose_ceye, distance_nose_cmouth, distance_eye, distance_mouth, l_eye, r_eye

def process_image(img):
    remember = 0
    rotate_img = 0
    import glob
    #detector = SCRFD(model_file='./det.onnx')
    # detector = SCRFD(model_file='/home/maicg/Documents/python-image-processing/insight-face/onnx/scrfd_500m.onnx')
    # detector = SCRFD(model_file='/home/maicg/Documents/python-image-processing/insight-face/onnx/scrfd_34g.onnx')
    detector = SCRFD(model_file='/home/maicg/Documents/python-image-processing/code-edit-insightFace/onnx/scrfd_2.5g_bnkps.onnx')
    detector.prepare(-1)
    
    h, w, c = img.shape
    # h = int(img.shape[0])
    # w = int(img.shape[1])
    # print(h,w)
    area_base = h*w
    # print("area===", area_base)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (640,640))

    for _ in range(1):
        ta = datetime.datetime.now()
        # bboxes, kpss = detector.detect(img, 0.5, input_size = (640, 640)) #max_num=1 thi ko phat hien chinh xac
        # bboxes, kpss = detector.detect(img, 0.5, input_size = (640, 640), max_num=1)
        bboxes, kpss = detector.detect(img, 0.65)
        tb = datetime.datetime.now()
    if kpss is not None:
        print("======================================")
        # print(kpss.shape)
        # print(kpss)
    tl = 0
    tl1 = 0
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1,y1,x2,y2,_ = bbox.astype(np.int)
        _,_,_,_,score = bbox.astype(np.float)
        # print("==================score", score)
        # cv2.rectangle(img, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
        crop_img = img[y1:y2, x1:x2]
        # h1, w1 = np.shape(crop_img)
        h1 = int(crop_img.shape[0])
        w1 = int(crop_img.shape[1])
        area_crop = h1*w1
        if kpss is not None:
            kps = kpss[i]
            distance12, distance_nose1, distance_nose2, distance_center_eye_mouth, distance_nose_ceye, distance_nose_cmouth, distance_eye, distance_mouth, l_eye, r_eye = xyz_coordinates(kps)
            if (distance_nose1-distance_nose2) <= 0:
                # print("=====================dt1,dt2",distance_nose1,distance_nose2)
                tl = distance_nose1/distance_nose2
            else: 
                # print("else=====================dt1,dt2",distance_nose1,distance_nose2)
                tl = distance_nose2/distance_nose1
            
            if (distance_nose_ceye - distance_nose_cmouth) <= 0:
                tl1 = distance_nose_ceye/distance_nose_cmouth
            else:
                tl1 = distance_nose_cmouth/distance_nose_ceye

            # print(tl)


            if area_crop == 0:
                break
            elif (area_base/area_crop) > ((1080*1920)/(64*64)):
                print("hinh nho")
            else:
                if distance12 >= distance_nose1 and distance12 >= distance_nose2:
                    if distance_center_eye_mouth >= distance_nose_ceye and distance_center_eye_mouth >= distance_nose_cmouth:
                        # if tl >= 0.6 and tl1 >= 0.6:
                            remember=1
                            rotate_img = alignment(crop_img, l_eye, r_eye)
                            # cv2.imwrite('./demo2/cr2/frameNEW{0}_{1}_{2}_{3}.jpg'.format(k, round(tl, 2), round(tl1, 2), round(score, 2)), crop_img)
                            # plt.imshow(rotate_img[:,:,::-1])
                            # plt.show()
    return rotate_img, remember
                        # else:
                        #     rotate_img = alignment(img, l_eye, r_eye)
                        #     # cv2.imwrite('./demo2/er2/frameNEW{0}_{1}_{2}_{3}.jpg'.format(k, round(tl, 2), round(tl1, 2), round(score, 2)), crop_img)
        
def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = cv2.cvtColor(filename, cv2.COLOR_BGR2RGB)
	# convert to array
	pixels = asarray(image)
	
	image = Image.fromarray(pixels)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

def get_normalized(face_array):
    # scale pixel values
    face_pixels = face_array.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    # std_adj = std.clamp(min=1.0/(float(face_pixels.numel())**0.5))
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    # samples = expand_dims(face_pixels, axis=0)

    return face_pixels

def computeEmb(img1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    img1 = extract_face(filename=img1)
    nor_img1 = get_normalized(img1)

    convert_tensor = transforms.ToTensor()
    conv_img1=convert_tensor(nor_img1)

    x_aligned1=[]
    x_aligned1.append(conv_img1)
    test_aligned1 = torch.stack(x_aligned1).to(device)
    test_embeddings1 = net(test_aligned1).detach().cpu()

    # net = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    # img_embedding1 = net(conv_img1.unsqueeze(0))

    return test_embeddings1


def computeCosin(emb1, img2):
    emb2 = computeEmb(img2)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(emb1, emb2)
    # print("goc ti le giua anh 1 va 2: ", output)
    return output


# def compare_image(img1, img2):
#     image1 = process_image(img1)
#     image2 = process_image(img2)

#     results = computeCosin(image1,image2)
#     # print("==========KQ run: ", results)
#     return results

# def main():
#     cam_port=1
#     # cap = cv2.VideoCapture(cam_port)
#     cap = cv2.VideoCapture('rtsp://ai_dev:123654789@@@192.168.15.10:554/Streaming/Channels/1601')
    
#     path = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/dataExper/dataDemo'
#     path_dir = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/dataExper/dataTest'
#     k=0
#     if cap.isOpened():
#         while True:
#             for i in range(20):
#                 result, img = cap.read()
#             for i in range(4):
#                 result, img = cap.read()
#             # plt.imshow(img[:,:,::-1])
#             # plt.show()
#             img_detect, remember = process_image(img)
#             count = 0
#             avr_time = 0
#             time_start = time.time()
#             predict=[]
#             label = []
#             if remember == 1:
#                 for image in os.listdir(path):
#                     pathName = [os.path.join(path,image)]
#                     for pathtest in pathName:
#                         img2 = cv2.imread(pathtest)
#                         img_origin, remember1 = process_image(img2)
#                         # print('done')
#                         result = computeCosin(img_origin, img_detect)
#                         print("===============", result)
#                         predict.append(result.item())
#                         label.append(pathtest)
#                         count = count + 1
#                 print('ket qua cuoi cung', max(predict))
#                 if max(predict) >= 0.65:
#                     # print("vi tri anr: ", label[predict.index(max(predict))])
#                     path_img = label[predict.index(max(predict))]
#                     path_img = path_img[-16:-4]
#                     print(path_img)
#                     index_label = path_img.find("/")
#                     directory = path_img[index_label+1:]
#                     print(directory)
#                     try:
#                         dir_fold = os.path.join(path_dir, directory)
#                         os.makedirs(dir_fold, exist_ok = True)
#                         frame_img_path = dir_fold + '/frame' + str(k) + '.jpg'
#                         print(frame_img_path)
#                         cv2.imwrite(frame_img_path, img_detect)
#                         print("Directory created successfully")
#                         k=k+1
#                     except OSError as error:
#                         print("Directory can not be created")
#                 else:
#                     print("unknow")
#                     try:
#                         dir_fold = os.path.join(path_dir, 'unknow')
#                         os.makedirs(dir_fold, exist_ok = True)
#                         frame_img_path = dir_fold + '/frame' + str(k) + '.jpg'
#                         print(frame_img_path)
#                         cv2.imwrite(frame_img_path, img_detect)
#                         k=k+1
#                     except OSError as error:
#                         print("Directory can not be created")
                
                
#                 time_end = time.time()
#                 avr_time = round(((time_end-time_start)/count), 2)
            
#                 print(avr_time)
#                 # print('Doneeeee')
#         cap.release()
#     cv2.destroyAllWindows()

# main()



