import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from audl_cv.game import Game
import cv2
from pathlib import Path
from typing import List, Tuple
import argparse

parser = argparse.ArgumentParser(description="Annotate clips.")
parser.add_argument(
    "game_url", metavar="U", type=str, nargs=1, help="URL to the game data"
)


class AnnotationGUI:
    def __init__(self, clip_path: str, possession_df: pd.DataFrame):
        self.locs = []
        self.wait_for_click = True
        self.clip_path = clip_path
        self.possession_df = possession_df
        self.frame_idx = 0

    # find these urls in all_games.txt
    def add_point(self, event):
        """ """
        x, y = event.x, event.y
        if event.inaxes:
            ax = event.inaxes  # the axes instance
            self.locs.append((self.frame_idx, event.xdata, event.ydata))
            ax.scatter(event.xdata, event.ydata, color="red")
            self.wait_for_click = False

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
        fig, axs = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": (3, 1)}, figsize=(14, 6)
        )
        plt.suptitle(self.clip_path)

        axs[1].plot(self.possession_df["x"], self.possession_df["y"])

        # Make it look like a field
        axs[1].set_xlim(-26.65, 26.65)
        axs[1].set_ylim(0, 120)
        axs[1].axhline(0, color="black", linewidth=1)
        axs[1].axhline(20, color="black", linewidth=1)
        axs[1].axhline(60, color="black", linewidth=1)
        axs[1].axhline(100, color="black", linewidth=1)
        axs[1].axhline(120, color="black", linewidth=1)

        cap = cv2.VideoCapture(self.clip_path)
        im = axs[0].imshow([[0, 0], [0, 0]], aspect="auto")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].get_yaxis().set_visible(False)
        axs[0].get_xaxis().set_visible(False)

        plt.connect("button_press_event", self.add_point)
        plt.ion()  # interactive plot

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                im.set_data(frame[..., ::-1])
            else:
                break
            if self.frame_idx % 10 == 0:  # This reduces frame rate
                while self.wait_for_click:
                    plt.pause(0.01)
                self.wait_for_click = True
            self.frame_idx += 1
        cap.release()
        plt.ioff()
        plt.close()
        return self.postprocess_annotations(self.locs, self.frame_idx)


def annotate_game_clips(game: Game) -> None:
    """
    Loops through clips in possession_to_video file for the game and
    prompts annoation if an annotation is not already found.
    """
    poss_to_vid = pd.read_csv(
        game.possession_to_video_path, index_col="possession_number"
    )

    for possession in poss_to_vid.itertuples():
        annotation_path = game.make_clip_annotation_path(
            possession.Index, possession.starttime, possession.endtime
        )
        if Path(annotation_path).is_file():
            print(
                f"Skipping annotation for possession: {possession.Index}."
                f" Annotation found at {annotation_path}"
            )
            pass
        elif possession.is_quality:
            clip_path = game.clip_possession(possession.Index)
            possession_df = game.load_possession_df(possession.Index, possession.home)

            gui = AnnotationGUI(clip_path, possession_df)
            annotations_df = gui.do_gui()
            annotations_df.to_feather(annotation_path)
        else:
            print(
                f"Skipping annotation for possession: {possession.Index}."
                " Possession marked as is_quality=False"
            )
            pass
    return


if __name__ == "__main__":
    args = parser.parse_args()
    g = Game(args.game_url[0])
    annotate_game_clips(g)
