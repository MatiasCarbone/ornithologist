import os, sys
import json
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

# Import module from another folder
sys.path.insert(0, 'src/features')
from preprocessing_functions import *

# Avoid tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# Singleton class: only one instance is allowed
class _Bird_Classifier(object):
    __instance = None

    def __new__(cls, model_path, json_path, sr=16000, window_length=1):
        if cls.__instance == None:
            print('\nInitializing instance of Bird Classifier...', end='  ')
            cls.__instance = super(_Bird_Classifier, cls).__new__(cls)
            cls.model = keras.models.load_model(model_path)
            cls.sr = sr
            cls.length = window_length

            with open(json_path) as j:
                data = json.load(j)
                cls.label_map = data['label_map']
            print('Done!')
        else:
            print('Bird Classifier was already initialized!')
        return cls.__instance

    def preprocess(self, audio_sample_path, hp=700, remove_silence=True):
        # Validate file extension
        if audio_sample_path.split('.')[-1] not in ['flac', 'wav', 'mp3']:
            raise ValueError('Invalid file extension!')

        # Load audio file
        y, sr = li.load(audio_sample_path, sr=self.sr, mono=True)
        # Apply high-pass filter
        y = apply_butter_highpass(data=y, cutoff=hp, fs=self.sr, order=5)

        # Delete silent sections
        if remove_silence:
            y, divisions = remove_silent_sections(y, thresh=20, hop=2048, return_splits=True)

        # Split into segments of desired length
        windows = split_audio_signal(y, target_length=self.length, samplerate=self.sr)

        return windows, divisions

    def predict(self, audio_sample_path):
        windows, divisions = self.preprocess(audio_sample_path)

        predictions = []
        for window, division in zip(windows, divisions):
            mfccs = li.feature.mfcc(y=window, sr=self.sr, n_mfcc=26, hop_length=512, n_fft=2048)
            mfccs = mfccs.transpose()
            mfccs = mfccs[np.newaxis, ..., np.newaxis]

            prob_dist = self.model.predict(mfccs, verbose=0).flatten()
            label = np.argmax(prob_dist)
            label_str = self.label_map[label]
            probability = round(prob_dist[label] * 100, 2)

            predictions.append((label, label_str, probability))

        return predictions


# This constants must match the parameter used for preprocessing MFCCs for training
SAMPLE_RATE = 16000
WINDOW_LENGTH = 1

TRAINED_MODEL_PATH = './models/00-production_model/production_model.keras'
AUDIO_FILE_PATH = './src/models/Test file (vanellus chillensis).flac'
JSON_PATH = './datasets/xeno_canto_birds/mfccs.json'

classifier = _Bird_Classifier(
    TRAINED_MODEL_PATH,
    JSON_PATH,
    sr=SAMPLE_RATE,
    window_length=WINDOW_LENGTH,
)

predictions = classifier.predict(AUDIO_FILE_PATH)
