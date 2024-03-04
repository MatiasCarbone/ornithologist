import os
import json
import librosa as li
from progress.bar import ChargingBar as Bar
from preprocessing_functions import *


# Preprocess raw audio files and extract MFCCs.
def preprocess_audio_dataset(
    audio_path, json_path=None, mfcc_count=26, hop=512, fft_len=2048, sr=16000, window_length=1
):
    data_dict = {
        'label_map': [],
        'encoded_labels': [],
        'mfccs': [],
        'files': [],
    }

    n_files = len([os.path.join(root, f) for root, dirs, files in os.walk(audio_path) for f in files])

    with Bar('Processing audio data', max=n_files) as bar:
        for i, (path, _, files) in enumerate(os.walk(audio_path)):
            if path == audio_path:  # Ignore parent folder
                continue

            # Add unique labels to label_map list
            label = path.split('/')[-1]
            if label not in data_dict['label_map']:
                data_dict['label_map'].append(label)

            for f in files:
                # Preprocess audio file and divide it into several windows
                windows = get_audio_windows(
                    os.path.join(path, f), sr=sr, length=window_length, hp=800, remove_silece=True
                )

                for window in windows:
                    # Add encoded label to encoded_labels list
                    index = data_dict['label_map'].index(label)
                    data_dict['encoded_labels'].append(index)

                    # Add original file path to files list
                    data_dict['files'].append(os.path.join(path, f))

                    # Compute MFCCs
                    mfccs = li.feature.mfcc(y=window, sr=sr, n_mfcc=mfcc_count, hop_length=hop, n_fft=fft_len)

                    # Append MFCCs to list. Casting np.array to list is required for saving as JSON file.
                    data_dict['mfccs'].append(mfccs.transpose().tolist())

                bar.next()

    # Store data dictionary in JSON file
    if json_path:
        with open(json_path, 'w') as jf:
            json.dump(data_dict, jf, indent=4)
            print(f'\n• Successfully saved preprocessed data to {json_path}!')
            file_count = len(data_dict['files'])
            print(f'\n• {file_count} audio samples were processed!\n')
    return data_dict


SAMPLE_RATE = 16000
WINDOW_LENGTH = 3

audio_path = './datasets/xeno_canto_birds/audio/'
json_path = './datasets/xeno_canto_birds/mfccs.json'

preprocess_audio_dataset(audio_path, json_path=json_path, window_length=3)
