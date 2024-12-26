
# sudo apt-get install tesseract-ocr
# pip install opencv-python
# pip install imutils
# pip install pytesseract

import cv2
import imutils
import pytesseract
from pytesseract import Output


def image_orientation_corrector(image_path):
    # Load the input image, convert it from BGR to RGB channel ordering,
    # and use Tesseract to determine the text orientation
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The image at path {image_path} could not be found.")
        
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image_to_osd function is used to extract the orientation detail of an image
    results = pytesseract.image_to_osd(rgb, config='--psm 0 -c min_characters_to_try=5',output_type=Output.DICT)
    
    # Display the orientation information
    print("[INFO] detected orientation: {}".format(results["orientation"]))
    print("[INFO] rotate by {} degrees to correct".format(results["rotate"]))
    print("[INFO] detected script: {}".format(results["script"]))
    
    # Rotate the image to correct the orientation
    rotated = imutils.rotate_bound(image, angle=results["rotate"])
    cv2.imwrite('orientation_corrected_image.jpg', rotated)

    # Show the original image and output image after orientation correction
    cv2.imshow("Original", image)
    cv2.imshow("Output", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# image_orientation_corrector("/home/lab-00/Downloads/quay2.jpg")
image_orientation_corrector("/home/lab-00/Downloads/orin3_q.jpg")
# image_orientation_corrector("/home/lab-00/Downloads/goc.jpg")

