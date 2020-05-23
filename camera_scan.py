# USAGE
# python camera_scan.py --image images/page.jpg

# import the necessary packages
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


def run():
    np.set_printoptions(precision=1)

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Our operations on the frame come here
        image_height = frame.shape[0]
        edged, overlay, warped = process_frame(frame, image_height)

        # Display the resulting frame
        frame = cv2.flip(frame, 1)
        cv2.imshow("Frame", frame)
        cv2.imshow("edged", edged)
        cv2.imshow("overlay", overlay)
        cv2.imshow("warped", imutils.resize(warped, height=512))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def process_frame(image, image_height=1000):
    ratio = image.shape[0] / image_height
    orig = image.copy()
    image = imutils.resize(image, height=image_height)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 75, 200)

    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    # sort the contour features by area from largest to smallest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screen_cnt = None

    # loop over the contours and try to find the largest 4-pointed polygon
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screen_cnt = approx
            break

    warped = np.random.rand(image.shape[0], image.shape[1])

    if screen_cnt is not None:
        cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 2)
        warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)
        # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # threshold = threshold_local(warped, 11, offset=10, method="gaussian")
        # warped = np.array(warped > threshold).astype("uint8") * 255

    return edged, image, warped


if __name__ == '__main__':
    run()
