from PIL import Image
import os

img = Image.open("/home/maicg/Documents/Me/CERBERUS/CerberusAISDK/database/face_25_4/person60/find_person_person60_0.510000_24.484400.jpg")
file_size = os.path.getsize("/home/maicg/Documents/Me/CERBERUS/CerberusAISDK/database/face_25_4/person60/find_person_person60_0.510000_24.484400.jpg") / 1024 # chuyển đổi kích thước sang kB
print("Dung lượng của ảnh là", file_size, "kB")