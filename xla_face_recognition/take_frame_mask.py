import cv2
import os 

path_dir_read = '/home/maicg/Downloads/dataFacerecog/face3/'
path_save_img = '/home/maicg/Downloads/dataFacerecog/khau_trang3/'

# # chuyen het anh ve duoi .jpg
# for dir in os.listdir(path_dir_read):
#     pathdir = os.path.join(path_dir_read,dir)
#     i = 0
#     for image in sorted(os.listdir(pathdir)):
#         try:
#             pathName = os.path.join(pathdir,image)
#             img2 = cv2.imread(pathName)
#             path_save = path_save_img + dir
#             os.makedirs(path_save,exist_ok=True)
#             path_save = path_save + '/' + str(i) + ".jpg"
#             print(path_save)
#             cv2.imwrite(path_save, img2)
#             i = i+ 1
#         except:
#             continue

#lay 1 frame trong anh
for dir in os.listdir(path_dir_read):
    pathdir = os.path.join(path_dir_read,dir)
    i = 0
    for image in sorted(os.listdir(pathdir)):
        pathName = os.path.join(pathdir,image)
        img2 = cv2.imread(pathName)
        path_save = path_save_img + dir
        os.makedirs(path_save,exist_ok=True)
        path_save = path_save + '/' + dir + ".jpg"
        print(path_save)
        cv2.imwrite(path_save, img2)
        i = i+ 1
        if i == 1:
            break