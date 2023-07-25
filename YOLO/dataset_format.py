
def txt_format_yolo(pil_bbox, width, height):
    xcenter = ((pil_bbox[0] + pil_bbox[2]) / 2) / width
    ycenter = ((pil_bbox[1] + pil_bbox[3]) / 2) / height
    w = (pil_bbox[2] - pil_bbox[0]) / width
    h = (pil_bbox[3] - pil_bbox[1]) / height
    return [xcenter, ycenter, w, h]