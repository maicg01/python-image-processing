def txt_format_yolo_pose(label, pil_bbox, landmarks, width, height):
    # file_name = os.path.basename(path).split('.')[0] //lay ten file trong duong dan /home/to/ten.jpg -> ten
    list_format = []
    xcenter = ((pil_bbox[0] + pil_bbox[2]) / 2) / width
    ycenter = ((pil_bbox[1] + pil_bbox[3]) / 2) / height
    w = (pil_bbox[2] - pil_bbox[0]) / width
    h = (pil_bbox[3] - pil_bbox[1]) / height
    list_format.append(label)
    list_format.append(xcenter)
    list_format.append(ycenter)
    list_format.append(w)
    list_format.append(h)

    for i in range(5):
        point_x = (landmarks[2 * i]) / width
        point_y = (landmarks[2 * i + 1]) / height
        list_format.append(point_x)
        list_format.append(point_y)
        list_format.append(2)
    
    my_string = ' '.join(map(str, list_format))

    return my_string