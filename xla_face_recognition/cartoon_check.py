#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union
import argparse

import cv2
import sys
import os


def is_cartoon(
    image: Union[str, Path],
    threshold: float = 0.9,
    preview: bool = False,
) -> bool:
    # read and resize image
    img = cv2.imread(str(image))
    img = cv2.resize(img, (1024, 1024))

    # blur the image to "even out" the colors
    color_blurred = cv2.bilateralFilter(img, 6, 250, 250)

    if preview:
        cv2.imshow("blurred", color_blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # compare the colors from the original image to blurred one.
    diffs = []
    for k, color in enumerate(("b", "r", "g")):
        # print(f"Comparing histogram for color {color}")
        real_histogram = cv2.calcHist(img, [k], None, [256], [0, 256])
        color_histogram = cv2.calcHist(color_blurred, [k], None, [256], [0, 256])
        diffs.append(
            cv2.compareHist(real_histogram, color_histogram, cv2.HISTCMP_CORREL)
        )

    print(sum(diffs)/3)
    return sum(diffs) / 3 > threshold


if __name__ == "__main__":

    path_folder = sys.argv[1]
    for path in os.listdir(path_folder):
        image = os.path.join(path_folder, path)
        if is_cartoon(image):
            print(f"{image} is a cartoon!")
        # else:
        #     print(f"{image} is a photo!")