# Test code modified from  https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
from __future__ import print_function
import cv2
import numpy as np


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.01
THRESHOLD = 0.8


def alignImages(im1, im2):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect SIFT features and compute descriptors.
    detector = cv2.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = detector.detectAndCompute(im2Gray, None)

    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Do 2NN heuristic
    print(len(raw_matches))
    matches = []
    for match_1, match_2 in raw_matches:
        # print(match_1.distance, match_2.distance)
        if match_1.distance < THRESHOLD * match_2.distance:
            matches.append(match_1)

    print(len(matches))

    # Sort matches by score
    list(matches).sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


if __name__ == "__main__":

    # Read reference image
    refFilename = "Group 2.jpeg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "BOS-NY-endzone.png"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)
