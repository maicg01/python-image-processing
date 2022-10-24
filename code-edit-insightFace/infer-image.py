from __future__ import division
import datetime
import imp
from re import S
import numpy as np
import onnx
import onnxruntime
import os
import os.path as osp
import cv2
import sys

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def distance2bbox(points, distance, max_shape=None):
    print("point 1: ", len(points[:, 0]))
    print("point 2: ", len(distance[:, 0]))
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
        print("khoi chay init")
        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        self.batched = False
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file) #kiem tra xem 1 duong dan co ton tai hay khong
            self.session = onnxruntime.InferenceSession(self.model_file, None) #tai va chay mot model
            print("session: ", self.session)
        self.center_cache = {}
        print("center_cache", self.center_cache)
        self.nms_thresh = 0.4
        print("nms_thresh", self.nms_thresh)
        self._init_vars()
        # print(self._init_vars())
    
    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0] #lay thong tin input dau vao
        print("input_cfg: ", input_cfg)
        input_shape = input_cfg.shape #lay kich thuoc input dau vao
        print("input_shape: ", input_shape)


        if isinstance(input_shape[2], str): #kiem tra input_shap[2] co la chuoi khong
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
            print("self.input_size ", self.input_size)
        input_name = input_cfg.name
        outputs = self.session.get_outputs()

        print("len outputs: ", len(outputs))
        print("outputs: ", outputs[0].shape)
        # print("len outputs: ", len(outputs[0].shape))

        
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        print("output_names: ", output_names)


        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1 #
        
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2 #point 1:  6400 -> point 1 va point 2 phai co cung len moi tru duoc
                                #point 2:  12800

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider']) #tang toc phan cung

    def detect(self, img, thresh=0.5, input_size = None, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
            
        im_ratio = float(img.shape[0]) / img.shape[1]
        # print("shape img", img.shape)

        model_ratio = float(input_size[1]) / input_size[0]
        # print("model_ratio: ", model_ratio)
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
        # print("scores_list: ", scores_list)

        scores = np.vstack(scores_list) #xep vao mot ngan sep
        # print("scores: ", scores)
        scores_ravel = scores.ravel() #thay doi mang nhieu chieu thanh mang phang
        # print("scores_ravel: ", scores_ravel)
        
        order = scores_ravel.argsort()[::-1] # tra ve cac chi so sap xep theo thu tu giam dan
        # print("order: ", order)
        bboxes = np.vstack(bboxes_list) / det_scale
        # if self.use_kps:
        #     kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False) #xếp chuỗi các mảng đầu vào theo chiều ngang (tức là theo cột) để tạo thành một mảng duy nhất
        # print("bboxes: ", bboxes)
        # print("scores: ", scores)
        # print("pre_det: ", pre_det)
        pre_det = pre_det[order, :] #sap xep theo thu tu order (giam dan)
        # print("pre_det: ", pre_det)

        keep = self.nms(pre_det)
        # print("keep: ", keep)

        det = pre_det[keep, :]
        print("===========================det: ", det)
        kpss = None
        return det, kpss
    
    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        # print("input_size: ", input_size)

        
        blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True) #preprocessing images va chuan bi du lieu cho phan loai trong model pretrained
                                                                                            # Mean subtraction, SCALing, channel swapping
        # print("blob: ", blob)

        net_outs = self.session.run(self.output_names, {self.input_name : blob})
        # print("net_outs: ", net_outs)
        # print("===========================================")
        # print("net_outs: ", len(net_outs))
        
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        # print("input_height: ", input_height)
        # print("input_width: ", input_width)
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            # print("idx, stride", idx, stride)
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                # print("===========================================")
                # print("scores: ", len(scores))
                bbox_preds = net_outs[idx + fmc][0] #fmc = len(net_outs) // 2
                # print("bbox_preds: ", bbox_preds)
                bbox_preds = bbox_preds * stride
                print("bbox_preds: ", len(bbox_preds))
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx] #net_outs = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32'] minh hoa theo ten score: trong so bbox: toa do hop gioi han
                bbox_preds = net_outs[idx + fmc] #neu lay o ind = 0 tuc la score_8 thi tuong ung hop du doan bbox_8 => fmc = 3
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
                # print("anchor_centers: ", anchor_centers)
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                    print("=============================================")
                    # print("pre anchor_centers: ", anchor_centers)
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores>=thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
        return scores_list, bboxes_list, kpss_list

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        print("===============scores: ", scores)

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1] #sap xep theo thu tu giam dan

        keep = []
        while order.size > 0:
            i = order[0] #gan i bang vi tri co trong so lon nhat
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

def main():
    import glob
    #detector = SCRFD(model_file='./det.onnx')
    detector = SCRFD(model_file='/home/maicg/Documents/python-image-processing/code-edit-insightFace/onnx/scrfd_2.5g_bnkps.onnx')
    detector.prepare(-1)
    img_paths = ['/home/maicg/Documents/python-image-processing/person.jpg']
    for img_path in img_paths:
        img = cv2.imread(img_path)

        for _ in range(1):
            ta = datetime.datetime.now()
            bboxes, kpss = detector.detect(img, 0.5, input_size = (640, 640))
            # bboxes, kpss = detector.detect(img, 0.5)

            tb = datetime.datetime.now()
            print('all cost:', (tb-ta).total_seconds()*1000)
        print(img_path, bboxes.shape)
        if kpss is not None:
            print(kpss.shape)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1,y1,x2,y2,score = bbox.astype(np.int)
            cv2.rectangle(img, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
            # if kpss is not None:
            #     kps = kpss[i]
            #     for kp in kps:
            #         kp = kp.astype(np.int)
            #         cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
        filename = img_path.split('/')[-1]
        print('output:', filename)
        cv2.imwrite('./outputs/%s'%filename, img)
        print('Doneeeee')

main()