# has_mask = False

# name = "mai_cg"
# if "mai" in name:
#     has_mask = True
# if has_mask:
#     print("Condition is True")
# else:
#     print("Condition is False")

# import numpy as np


# start_row = 1
# end_row = 3
# start_col = 4
# end_col = 6
# mt = np.random.randint(0, 10, (8, 8))

# print(mt)
# print(mt[start_row:end_row+1, start_col:end_col+1])


# import numpy as np

# # Tạo một mảng numpy chứa các tên
# names = np.array(["Mai", "Hoa", "Nam", "Mai", "An"])

# # Đổi tất cả các tên "Mai" thành tên "Nam"
# names[names == "Mai"] = "Nam"

# # In mảng tên sau khi thay đổi
# print(names)


from shapely.geometry import Polygon, box
import cv2
import numpy as np
import json
image_path = "/home/maicg/Downloads/3542de27-102f-498d-ad56-4da446b05b53.jpeg"
img = cv2.imread(image_path)
img = cv2.resize(img, (2080,2080))

# Bounding box cần kiểm tra
x1, y1, x2, y2 = 900, 400, 1000, 410  # Thay thế bằng giá trị thực tế

# Chuyển bounding box thành đối tượng Polygon
bounding_box_polygon = box(x1, y1, x2, y2)
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

def scale_polygon(polygon, original_size, new_size):
    scale_width = new_size[0] / original_size[0]
    scale_height = new_size[1] / original_size[1]
    
    scaled_polygon = [[int(point[0] * scale_width), int(point[1] * scale_height)] for point in polygon]
    return scaled_polygon

def process_polygon_dict(image, dict_polygon):
    for item in dict_polygon:
        polygon = eval(item['polygon'])
        scaled_polygon = scale_polygon(polygon, (1920,1080), (image.shape[1], image.shape[0]))
        # Update the item with the new scaled polygon
        item['polygon'] = str(scaled_polygon)
    
    return dict_polygon

# Danh sách polygons từ dict_polygon
dict_polygon = [
    {
        "id": 14,
        "polygonType": "VIP",
        "polygon": "[[1140, 69], [1266, 135], [1785, 357], [831, 288]]",
        "line": "[]",
        "name": "V",
        "active": True,
        "cameraId": 1
    },
    {
        "id": 15,
        "polygonType": "PRESS",
        "polygon": "[[852, 180], [1185, 195], [972, 498], [396, 423]]",
        "line": "[]",
        "name": "p",
        "active": True,
        "cameraId": 1
    }
]

dict_polygon = process_polygon_dict(img, dict_polygon)
print(dict_polygon)

# dict_polygon = []

if not dict_polygon or dict_polygon is None:
    print("dict_polygon rỗng")
else:
    for item in dict_polygon:
        # Lấy polygon từ dictionary và chuyển nó thành đối tượng Polygon
        polygon_points = eval(item['polygon'])  # Chú ý: sử dụng eval có rủi ro về bảo mật, chỉ nên sử dụng với dữ liệu đáng tin cậy
        print("polygon_points: ", type(polygon_points))
        polygon = Polygon(polygon_points)

        polygon_points = np.array(polygon_points, np.int32)
        # polygon_points = polygon_points.reshape((-1, 1, 2))  # Cần reshape cho phù hợp với OpenCV
        cv2.polylines(img, [polygon_points], True, (255, 0, 0), 2)
        print("polygon_points[0]: ", polygon_points[0])
        cv2.putText(img, "THU", (polygon_points[0][0],polygon_points[0][1] - 10) , cv2.FONT_HERSHEY_SIMPLEX, 2, (255,140,0), 4, cv2.LINE_AA)
        
        # Hiển thị ảnh
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Kiểm tra xem bounding box có nằm trong polygon này không
        if polygon.contains(bounding_box_polygon):
            print(f"The bounding box is inside polygon {item['id']} of type {item['polygonType']}")
            break
        else:
            print("The bounding box is not inside any of the polygons.")


        
# mask = np.zeros_like(img)
# result = img.copy()
# for item in dict_polygon:
#     # Lấy polygon từ dictionary và chuyển nó thành đối tượng Polygon
#     polygon_points = eval(item['polygon'])  # Chú ý: sử dụng eval có rủi ro về bảo mật, chỉ nên sử dụng với dữ liệu đáng tin cậy
#     print("polygon_points: ", type(polygon_points))
#     polygon = Polygon(polygon_points)
#     polygon_points = np.array(polygon_points, np.int32)

#     cv2.fillPoly(mask, [polygon_points], (255, 255, 255))
#     result = cv2.bitwise_and(img, mask)


# cv2.imshow('to den', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
