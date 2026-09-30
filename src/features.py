import librosa
import numpy as np
import os
from tqdm import tqdm

def extract_mfcc(file_path, max_len=150):
    # Load audio file 
    audio, sample_rate = librosa.load(file_path, duration=3.0, offset=0.5)
    
    # Extract 40 MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
    # Pad or truncate the sequence to a fixed length
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
        
    return mfcc.T 

def load_ravdess_data(data_path="data/"):
    X, y = [], []
    
    # Find all .wav files in the directory and subdirectories
    wav_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
                
    if not wav_files:
        raise ValueError("No .wav files found! Check that your Actor folders are inside the 'data/' folder.")

    print(f"Found {len(wav_files)} audio files. Extracting features...")
    
    # Loop through files with a progress bar
    for file_path in tqdm(wav_files):
        filename = os.path.basename(file_path)
        
        # RAVDESS naming: 03-01-[Emotion]-01-01-01-01.wav
        parts = filename.split('-')
        
        if len(parts) == 7:
            # Get the emotion label (convert to integer)
            # We subtract 1 because Neural Networks need labels to start at 0 (0 to 7)
            emotion_label = int(parts[2]) - 1 
            
            # Extract audio features
            mfcc = extract_mfcc(file_path)
            
            X.append(mfcc)
            y.append(emotion_label)
            
    return np.array(X), np.array(y)