import librosa
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from pydub import AudioSegment
import os

def preprocess_audio(audio_file, sr=16000):
    # Загрузка аудиофайла и преобразование его в моно-звук
    audio, _ = librosa.load(audio_file, sr=sr)
    # Удаление тишины из аудиофайла
    audio = librosa.effects.remix(audio, intervals=librosa.effects.split(audio))
    return audio


def extract_features(y, sr=16000, n_mfcc=20, hop_length=512):
    # Извлечение MFCC признаков из аудиофайла
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc.T


def cluster_features(features, n_clusters=2):
    # Кластеризация признаков с помощью KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    return kmeans


def get_speaker_timings(labels, hop_length=512, sr=16000):
    # Определение временных интервалов для каждого спикера
    speaker_timings = {}
    current_speaker = labels[0]
    start_time = 0

    for i, label in enumerate(labels[1:]):
        if label != current_speaker:
            end_time = (i * hop_length) / sr
            if current_speaker not in speaker_timings:
                speaker_timings[current_speaker] = []
            speaker_timings[current_speaker].append((start_time, end_time))
            start_time = end_time
            current_speaker = label

    end_time = (len(labels) * hop_length) / sr
    if current_speaker not in speaker_timings:
        speaker_timings[current_speaker] = []
    speaker_timings[current_speaker].append((start_time, end_time))

    return speaker_timings


def diarize_audio(audio_file, n_clusters=2):
    # Диаризация аудиофайла
    audio = preprocess_audio(audio_file)
    features = extract_features(audio)
    kmeans = cluster_features(features, n_clusters)
    speaker_timings = get_speaker_timings(kmeans.labels_)

    return speaker_timings

audio_file = 'input/Audio.ogg'
n_clusters = 2  # количество собеседников
speaker_timings = diarize_audio(audio_file, n_clusters)

# for speaker, timings in speaker_timings.items():
#     print(f"Спикер {speaker + 1}:")
#     for start, end in timings:
#         print(f"  {start:.2f} - {end:.2f} сек")