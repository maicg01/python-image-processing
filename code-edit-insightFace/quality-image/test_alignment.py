from facenetPreditctFunction import alignment, SCRFD, process_image_package, xyz_coordinates
import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    #load moder
    detector = SCRFD(model_file='/home/maicg/Documents/python-image-processing/code-edit-insightFace/onnx/scrfd_2.5g_bnkps.onnx')
    detector.prepare(-1)

    img = cv2.imread('/home/maicg/Documents/python-image-processing/AH-team.jpg')

    bboxes, kpss = process_image_package(img, detector)

    for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                x1,y1,x2,y2,_ = bbox.astype(np.int)
                _,_,_,_,score = bbox.astype(np.float)

                crop_img = img[y1:y2, x1:x2]
                crop_img = cv2.resize(crop_img, (112,112))
                plt.imshow(crop_img[:,:,::-1])
                plt.show()
                
                h1 = int(crop_img.shape[0])
                w1 = int(crop_img.shape[1])
                area_crop = h1*w1
                if kpss is not None:
                    kps = kpss[i]
                    distance12, distance_nose1, distance_nose2, distance_center_eye_mouth, distance_nose_ceye, distance_nose_cmouth, distance_eye, distance_mouth, l_eye, r_eye = xyz_coordinates(kps)


                    rotate_img = alignment(crop_img, l_eye, r_eye)
                    rotate_img = cv2.resize(rotate_img, (112,112))
                    plt.imshow(rotate_img[:,:,::-1])
                    plt.show()

main()