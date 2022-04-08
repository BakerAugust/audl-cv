from typing import Any, Tuple
import audl_advanced_stats as audl
import pandas as pd
from pathlib import Path
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


YOUTUBE_FILE = "youtube/youtube_urls.csv"
YOUTUBE_DIR = "youtube"
POSSESSION_TO_VIDEO_DIR = "possession_to_video"
POSSESSSION_CLIPS_DIR = "possession_clips"


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

        return df

    def clip_possession(self, possession_number: int) -> str:
        """
        Trys to clip full game by possession. Raises errors if the game is not yet downloaded or
        possession_to_video annotations cannot be found.
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
        poss_to_vid = pd.read_csv(
            self.possession_to_video_path, index_col="possession_number"
        )
        starttime = poss_to_vid.loc[possession_number, "start_time"]
        endtime = poss_to_vid.loc[possession_number, "end_time"]
        save_file = f"{self.data_path}/{POSSESSSION_CLIPS_DIR}/{self.get_game_name()}-{str(possession_number)}-{str(starttime)}-{str(endtime)}.mp4"

        if Path(save_file).is_file():
            pass
        else:
            ffmpeg_extract_subclip(
                self.youtube_file, starttime, endtime, targetname=save_file
            )

        return save_file
