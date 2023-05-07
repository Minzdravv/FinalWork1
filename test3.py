import librosa
import librosa
import IPython
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns


audio_data = './input/audio_1.wav'
y, sr = librosa.load(audio_data)
print(y, sr)
# Spectral Centroid
cent = librosa.feature.spectral_centroid(y=y, sr=sr)
plt.figure(figsize=(15,5))
plt.subplot(1, 1, 1)
plt.semilogy(cent.T, label='Spectral centroid')
plt.ylabel('Hz')
plt.xticks([])
plt.xlim([0, cent.shape[-1]])
plt.legend()
#
# import IPython.display as ipd
# plt.figure(figsize=(14, 5))
# librosa.display.waveshow(y, sr=sr)
# ipd.Audio(audio_data)
#
# # Seperation of Harmonic and Percussive Signals
# y_harmonic, y_percussive = librosa.effects.hpss(y)
# plt.figure(figsize=(15, 5))
# librosa.display.waveshow(y_harmonic, sr=sr, alpha=0.25)
# librosa.display.waveshow(y_percussive, sr=sr, color='r', alpha=0.5)
# plt.title('Harmonic + Percussive')
#
# # Beat Extraction
# tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)
# print('Detected Tempo: '+str(tempo)+ ' beats/min')
# beat_times = librosa.frames_to_time(beat_frames, sr=sr)
# beat_time_diff=np.ediff1d(beat_times)
# beat_nums = np.arange(1, np.size(beat_times))
#
# fig, ax = plt.subplots()
# fig.set_size_inches(15, 5)
# ax.set_ylabel("Time difference (s)")
# ax.set_xlabel("Beats")
# g=sns.barplot(beat_nums, beat_time_diff, palette="BuGn_d",ax=ax)
# g=g.set(xticklabels=[])