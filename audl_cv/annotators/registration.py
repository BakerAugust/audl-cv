import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from audl_cv.game import Game
import cv2
from pathlib import Path
from typing import List, Tuple
import argparse
from matplotlib.backend_bases import MouseButton

# Crop out the score bar in videos
YCROP = 590

parser = argparse.ArgumentParser(description="Annotate clips.")
parser.add_argument(
    "game_url", metavar="U", type=str, nargs=1, help="URL to the game data"
)


class RegistrationGUI:
    def __init__(self, clip_path: Path, possession_df: pd.DataFrame):
        self.matches = []  # Dict of frame idx: [(im_point, top_point),...]
        self.im_locs = []  # Points on image plane
        self.top_locs = []  # Points on top plane
        self.wait_for_click = True
        self.clip_path = clip_path
        self.possession_df = possession_df
        self.frame_idx = 0
        self.done = False  # Controlled by "x" key

    # find these urls in all_games.txt
    def add_point(self, event):
        """ """
        # If I wanted to add a line connecting...
        # https://stackoverflow.com/questions/17543359/drawing-lines-between-two-plots-in-matplotlib
        x, y = event.x, event.y
        print(x, y)
        print(event.xdata, event.ydata)
        if event.inaxes and event.button == MouseButton.RIGHT:
            ax = event.inaxes  # the axes instance
            ax.scatter(event.xdata, event.ydata, color="red")
            if ax == self.axs[0]:
                self.im_locs.append(  # process to img coords
                    (
                        (event.xdata + 0.5) / 2 * self.x_dim,
                        (event.ydata + 0.5) / 2 * self.y_dim,
                    )
                )
            elif ax == self.axs[1]:
                self.top_locs.append((event.ydata, event.xdata))

    def key_event(self, event):
        if event.key == "right":
            self.wait_for_click = False
        elif event.key == "x":
            self.done = True
        else:
            pass

    @staticmethod
    def postprocess_annotations(annotations: List[Tuple], n_frames: int) -> np.ndarray:
        """
        Does interpolation that enables lower frame rate annotation
        """
        arr = np.full(shape=(n_frames, 2), fill_value=np.nan)
        for idx, x, y in annotations:
            arr[idx, :] = (x, y)
        df = pd.DataFrame(arr).interpolate()
        df.columns = ["x", "y"]
        df = df.assign(frame_number=df.index)
        return df

    def do_gui(self):
        self.fig, axs = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": (3, 1)}, figsize=(14, 6)
        )
        self.axs = axs
        plt.suptitle(self.clip_path)

        axs[1].plot(self.possession_df["x"], self.possession_df["y"])

        # Make it look like a field
        axs[1].set_xlim(-26.65, 26.65)
        axs[1].set_ylim(0, 120)
        axs[1].get_yaxis().set_visible(False)
        axs[1].get_xaxis().set_visible(False)
        # hash marks
        for y in [-13.65, 13.65]:
            axs[1].axvline(y, color="grey", dashes=(1, 1))
        # major hlines
        for x in [0, 20, 60, 100, 120]:
            axs[1].axhline(x, color="black", linewidth=1)
        # minor hlines
        for x in [
            5,
            10,
            15,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
            105,
            110,
            115,
        ]:
            axs[1].axhline(x, color="grey", linewidth=0.5)
        print(self.clip_path)
        cap = cv2.VideoCapture(str(self.clip_path))
        im = axs[0].imshow([[0, 0], [0, 0]], aspect="auto")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].get_yaxis().set_visible(False)
        axs[0].get_xaxis().set_visible(False)

        plt.connect("button_press_event", self.add_point)
        plt.connect("key_press_event", self.key_event)
        plt.ion()  # interactive plot

        while cap.isOpened() and not self.done:
            ret, frame = cap.read()
            if ret:
                cropped_frame = frame[:YCROP, :, :]
                self.y_dim = cropped_frame.shape[0]
                self.x_dim = cropped_frame.shape[1]
                im.set_data(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            else:
                break
            if self.frame_idx % 10 == 0:  # This reduces frame rate
                while self.wait_for_click:
                    plt.pause(0.01)
                self.wait_for_click = True
            if self.im_locs:
                for im_loc, top_loc in zip(self.im_locs, self.top_locs):
                    self.matches.append(
                        {
                            "clip": str(self.clip_path),
                            "frame": self.frame_idx,
                            "im_x": im_loc[0],
                            "im_y": im_loc[1],
                            "top_x": top_loc[0],
                            "top_y": top_loc[1],
                        }
                    )

            self.frame_idx += 1
            self.im_locs = []  # Points on image plane
            self.top_locs = []  # Points on top plane
        cap.release()
        plt.ioff()
        plt.close()
        return self.matches


def annotate_game_clips(game: Game) -> None:
    """
    Loops through clips in possession_to_video file for the game and
    prompts annoation if an annotation is not already found.
    """
    poss_to_vid = pd.read_csv(
        game.possession_to_video_path, index_col="possession_number"
    )
    annotations = []
    poss_id = 1
    for possession in poss_to_vid.itertuples():
        if possession.is_quality:
            annotation_path = game.make_clip_annotation_path(
                possession.Index, possession.starttime, possession.endtime
            )
            if Path(annotation_path).is_file():
                print(
                    f"Skipping annotation for possession: {possession.Index}."
                    f" Annotation found at {annotation_path}"
                )
                pass

            else:
                clip_path = game.clip_possession(possession.Index)
                possession_df = game.load_possession_df(
                    possession.Index, possession.home
                )

                gui = RegistrationGUI(clip_path, possession_df)
                annotations += gui.do_gui()
        poss_id += 1
    df = pd.DataFrame(annotations)
    df.to_feather(str(g.make_homography_annotation_path()))
    return


if __name__ == "__main__":
    args = parser.parse_args()
    g = Game(args.game_url[0])
    out = annotate_game_clips(g)
    print(out)
