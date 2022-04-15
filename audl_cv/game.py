import audl_advanced_stats as audl
import pandas as pd
from pathlib import Path
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


YOUTUBE_FILE = "youtube/youtube_urls.csv"
YOUTUBE_DIR = "youtube"
POSSESSION_TO_VIDEO_DIR = "possession_to_video"
POSSESSSION_CLIPS_DIR = "possession_clips"
POSSESSSION_ANNOTATIONS_DIR = "possession_annotations"


class Game(audl.Game):
    """
    Extension of audl.Game base class to include video scraped from youtube.
    """

    def __init__(
        self,
        game_url,
        youtube_url=None,
        year=audl.constants.CURRENT_YEAR,
        data_path="data",
        upload=False,
        download=False,
    ):

        # Initial args for base class
        super().__init__(game_url, year, data_path, upload, download)

        if not youtube_url:
            # Try to get from file
            youtube_urls = pd.read_csv(
                f"{self.data_path}/{YOUTUBE_FILE}", index_col="external_game_id"
            )
            youtube_url = youtube_urls.loc[self.get_game_name(), "url"]

        self.youtube_url = youtube_url
        self.youtube_file = f"{self.data_path}/{YOUTUBE_DIR}/{self.get_game_name()}.mp4"
        self.possession_to_video_path = (
            f"{self.data_path}/{POSSESSION_TO_VIDEO_DIR}/{self.get_game_name()}.csv"
        )

    def download_video(self, override: bool = False) -> None:
        """
        Downloads video if not exists

        Parameters
        ---------
        override: bool = False
            Whether or not to download file again even if it exists.
        """
        if Path(self.youtube_file).is_file() and not override:
            print(f"Video already exists at {self.youtube_file}")
            return
        else:
            yt = YouTube(self.youtube_url)
            yt.streams.get_highest_resolution().download(filename=self.youtube_file)

    def load_possession_df(
        self, possession_number: int, home: bool = None
    ) -> pd.DataFrame:
        """
        Loads possession dataframe. Ripped code from audl.Game
        """

        if home:
            events = self.get_home_events(qc=False)
        else:
            events = self.get_away_events(qc=False)
        df = (
            events.query("t==[10, 12, 13, 20]")
            .query(f"possession_number=={possession_number}")
            .reset_index(drop=True)
            .reset_index()
            .assign(
                event=lambda x: x["index"] + 1,
                x=lambda x: x["x"],
                x_after=lambda x: x["x_after"],
                xyards_raw=lambda x: x["xyards_raw"],
            )
            .drop(columns=["index"])
            .copy()
        )
        # Re-label first event
        df.loc[df["event"] == 1, "event_name"] = "Start of Possession"
        df.loc[df["event"] == 1, "t"] = 0

        # Draw possession if there's data, otherwise draw a blank field
        last_row = df.loc[df["event"] == df["event"].max()].iloc[0].copy()

        # Add row for last event
        df = df.append(
            pd.Series(
                {
                    "x": last_row["x_after"],
                    "y": last_row["y_after"],
                    "t": last_row["t_after"],
                    "yyards_raw": last_row["yyards_raw"],
                    "xyards_raw": last_row["xyards_raw"],
                    "yards_raw": last_row["yards_raw"],
                    "play_description": last_row["play_description"],
                    "event_name": last_row["event_name_after"],
                    "event": last_row["event"] + 1,
                    "r": last_row["r_after"],
                }
            ),
            ignore_index=True,
        )

        return df

    def make_clip_path(
        self, possession_number: int, starttime: int, endtime: int
    ) -> str:
        return f"{self.data_path}/{POSSESSSION_CLIPS_DIR}/{self.get_game_name()}-{str(possession_number)}-{str(starttime)}-{str(endtime)}.mp4"

    def make_clip_annotation_path(
        self, possession_number: int, starttime: int, endtime: int
    ) -> str:
        return f"{self.data_path}/{POSSESSSION_ANNOTATIONS_DIR}/{self.get_game_name()}-{str(possession_number)}-{str(starttime)}-{str(endtime)}.feather"

    def load_possession_to_video(self) -> pd.DataFrame:
        """
        Load file that provides video start/end times for each possession.
        """
        if not Path(self.youtube_file).is_file():
            raise FileExistsError(
                """
                Full game file not found! Try running Game.download_video() first.
            """
            )

        if not Path(self.possession_to_video_path).is_file():
            raise FileExistsError(
                """
                possession_to_video annotations file not found at {self.possession_to_video_path}. Add annotation file
                to clip videos. 
            """
            )

        # Try to load possession_to_video
        return pd.read_csv(self.possession_to_video_path, index_col="possession_number")

    def clip_possession(self, possession_number: int) -> str:
        """
        Trys to clip full game by possession. Raises errors if the game is not yet downloaded or
        possession_to_video annotations cannot be found.
        """
        poss_to_vid = self.load_possession_to_video()

        starttime = poss_to_vid.loc[possession_number, "starttime"]
        endtime = poss_to_vid.loc[possession_number, "endtime"]
        save_file = self.make_clip_path(possession_number, starttime, endtime)

        if Path(save_file).is_file():
            pass
        else:
            ffmpeg_extract_subclip(
                self.youtube_file, starttime, endtime, targetname=save_file
            )

        return save_file

    def load_annotation(self, possession_number: int) -> pd.DataFrame:
        """
        Tries to read annotation file.
        """
        poss_to_vid = self.load_possession_to_video()
        starttime = poss_to_vid.loc[possession_number, "starttime"]
        endtime = poss_to_vid.loc[possession_number, "endtime"]
        path = self.make_clip_annotation_path(possession_number, starttime, endtime)

        return pd.read_feather(path)

    def annotated_clips(self) -> pd.DataFrame:
        """
        Returns a dataframe of which clips are downloaded with annotations.

        Clips the possessions if not done already.
        """
        poss_to_vid = self.load_possession_to_video()
        out = []
        for possession in poss_to_vid.itertuples():
            if possession.is_quality:
                clip_path = self.clip_possession(possession.Index)
                annotation_path = self.make_clip_annotation_path(
                    possession.Index, possession.starttime, possession.endtime
                )
                out.append(
                    {
                        "possession_number": possession.Index,
                        "clip_path": clip_path,
                        "has_annotation": Path(annotation_path).is_file(),
                    }
                )

        return pd.DataFrame(out)
