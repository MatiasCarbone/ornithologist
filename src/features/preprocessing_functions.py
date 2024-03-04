from scipy import signal
import numpy as np
import librosa as li


# Generate butterworth highpass coefficients
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


# Apply filter to signal
def apply_butter_highpass(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


# Remove sections of silence or low intensity signal
def remove_silent_sections(signal, thresh=20, hop=2048, return_splits=False):
    splits = li.effects.split(y=signal, top_db=thresh, frame_length=(hop * 2), hop_length=hop)

    stripped_audio = []

    for s in splits:
        split = signal[s[0] : s[1]]
        stripped_audio.extend(split)

    if return_splits:
        return np.asarray(stripped_audio), split
    else:
        return np.asarray(stripped_audio)


# Split audio into segments of desired length
def split_audio_signal(signal, target_length, samplerate):
    duration = li.get_duration(y=signal, sr=samplerate)
    n_windows = np.ceil(duration / target_length)
    audio_windows = []

    for n in range(int(n_windows)):
        s = signal[samplerate * n * target_length : samplerate * (n + 1) * target_length]

        if len(s) < target_length * samplerate:
            s = np.pad(s, (0, target_length * samplerate - len(s)), 'constant')

        audio_windows.append(s)

    return audio_windows


# Preprocess audio data and get preprocessed audio windows
def get_audio_windows(path, sr, length, hp=700, remove_silece=True):
    # Load audio file
    y, sr = li.load(path, sr=sr, mono=True)
    # Apply high-pass filter
    y = apply_butter_highpass(data=y, cutoff=hp, fs=sr, order=5)

    if remove_silece:
        # Delete silent sections
        y = remove_silent_sections(y, thresh=20, hop=2048)

    # Split into segments of desired length
    audio_segments = split_audio_signal(y, target_length=length, samplerate=sr)

    return audio_segments
