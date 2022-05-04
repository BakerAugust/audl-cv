from turtle import shape
from matplotlib import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from audl_cv.game import Game
import cv2
from pathlib import Path
import argparse

OUTPUT_SIZE = (590, 1280)
YCROP = 590


def warp2surface(img, im_pts, top_pts):
    """ """

    h, status = cv2.findHomography(im_pts, top_pts)

    img_surface = cv2.warpPerspective(
        img,
        M=h,
        dsize=(OUTPUT_SIZE[0], OUTPUT_SIZE[1]),
    )
    return img_surface


if __name__ == "__main__":
    imgs = []
    df = pd.read_feather("data/homography_annotations/2021-08-28-DAL-SD.feather")
    df = df.assign(
        top_x=(df["top_y"] + 26.65) * 10,  # blow up
        top_y=(120 - df["top_x"]) * 10,  # blow up
    )
    for c_idx, clip_df in df.groupby("clip"):
        for f_idx, frame_df in clip_df.groupby("frame"):
            if f_idx > 0:
                im_pts = []
                top_pts = []
                for _, row in frame_df.iterrows():
                    im_pts.append([row["im_x"], row["im_y"]])
                    top_pts.append([row["top_x"], row["top_y"]])

                cap = cv2.VideoCapture(row["clip"])

                # Sets frame to the one we want
                cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])

                ret, frame = cap.read()
                try:
                    warped = warp2surface(
                        frame[:YCROP, :, :], np.array(im_pts), np.array(top_pts)
                    )
                    imgs.append(warped)
                except Exception as e:
                    print(e)
                    pass

                cap.release()
                cv2.destroyAllWindows()

    base = np.zeros([OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3])
    i = 0
    for img in imgs:
        cv2.imwrite(f"temp_{i}.jpg", img)
        i += 1
        base = np.where(base == 0, img, base)

    cv2.imwrite("field_reference.jpg", base)
    # plt.imshow(base.astype("int"))
    # plt.savefig("field_reference.png")
