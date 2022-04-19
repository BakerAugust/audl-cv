import glob
import os

def collect_data_paths(data_dir_path):    
    annotation_path = data_dir_path.split('\\') + ['possession_annotations', '*.feather']
    annotation_path = os.path.join(*annotation_path)
    
    annotations = []
    
    for file in glob.glob(annotation_path):
        annotations.append(file)
    
    # clips = []
    # clip_path = data_dir_path.split('\\') + ['possession_clips', '*.mp4']
    # clip_path = os.path.join(*clip_path)
    # for game in annotations:
    #     clip = game.replace('possession_annotations', 'possession_clips').replace('.feather', '.mp4')
    #     clips.append(clip)
        
    return annotations

if __name__ == '__main__':
    print('hello world')
    
    annotations = collect_data_paths('data')
    
    print(annotations)
        