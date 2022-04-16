# audl-cv

Computer vision for AUDL

## Recommended Environment

Create virtualenv
```
pyenv virtualenv 3.9.6 audl-cv
```

Install pip packages
```
pip install -r requirements.txt
```

Install ffmpeg for moviepy. 
- On Mac, `brew install ffmpeg`
- On windows, try [ffmpeg downloads](https://ffmpeg.org/download.html) page.

## How the game videos are annotated
- Full videos are downloaded from YouTube.
- Videos are manually reviewed to assign start/end times that correspond to the `possession_number` of each possession in the game. There are also given an `is_qualty: bool` attribute which describes if they are suitable for CV training. These data are stored in `data/possession_to_video` and can be loaded through the `Game.load_possession_to_video()` method.
- Each clip is reviewed along with the AUDL possession maps (not time-bound) and top-view annotations are added for some down-sampled subset of frames (e.x. 1 fps). 
- Downsampled annotations are applied to each frame through linear interpolation.
- Annotations are stored in `data/possession_annotaions` and can be accessed through the `Game.load_annotation()` method.