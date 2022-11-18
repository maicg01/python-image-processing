import cv2
import os
import time
import pickle

from facenetPreditctFunction import process_image, computeEmb, computeCosin, fixed_image_standardization


def main():
    imageOriginSet = []
    labelOriginSet = []
    with open('X.pkl', 'rb') as f:
        imageOriginSet = pickle.load(f)

    with open('y.pkl', 'rb') as f:
        labelOriginSet = pickle.load(f)

    # cam_port=1
    # cap = cv2.VideoCapture(cam_port)
    # cap = cv2.VideoCapture('rtsp://ai_dev:123654789@@@192.168.15.10:554/Streaming/Channels/1601')
    cap = cv2.VideoCapture('/home/maicg/Documents/python-image-processing/video_gt.avi')
    
    path = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/dataExper/dataDemo'
    path_dir = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/dataExper/dataTest'
    k=0
    if cap.isOpened():
        while True:
            for i in range(4):
                result, img = cap.read()
            # plt.imshow(img[:,:,::-1])
            # plt.show()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_detect, remember = process_image(img)
            avr_time = 0
            
            predict=[]
            label = []
            if remember == 1:
                count = 0
                time_start = time.time()
                for embOrigin in imageOriginSet:
                    # print('done')
                    result = computeCosin(embOrigin, img_detect)
                    print("===============", result)
                    predict.append(result.item())
                    label.append(labelOriginSet[count])
                    count = count + 1
                print('ket qua cuoi cung', max(predict))
                if max(predict) >= 0.65:
                    
                    # print("vi tri anr: ", label[predict.index(max(predict))])
                    directory = label[predict.index(max(predict))]
                    print(directory)
                    try:
                        dir_fold = os.path.join(path_dir, directory)
                        os.makedirs(dir_fold, exist_ok = True)
                        frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(max(predict), 2))  + '.jpg'
                        print(frame_img_path)
                        # img_save = cv2.resize(img_detect, (160,160))
                        cv2.imwrite(frame_img_path, img_detect)
                        print("Directory created successfully")
                        k=k+1
                    except OSError as error:
                        print("Directory can not be created")
                else:
                    print("unknow")
                    try:
                        dir_fold = os.path.join(path_dir, 'unknow')
                        os.makedirs(dir_fold, exist_ok = True)
                        frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(max(predict), 2))  + '.jpg'
                        print(frame_img_path)
                        # img_save = cv2.resize(img_detect, (160,160))
                        cv2.imwrite(frame_img_path, img_detect)
                        k=k+1
                    except OSError as error:
                        print("Directory can not be created")
                
                
                time_end = time.time()
                avr_time = round(((time_end-time_start)/count), 2)
            
                print(avr_time)
                # print('Doneeeee')
        cap.release()
    cv2.destroyAllWindows()

main()