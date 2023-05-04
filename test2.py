import numpy as np
import librosa
from matplotlib import pyplot as plt
import sklearn
#pip install sckit-learn
#pip install sckit-datasets

audio_data = './input/Audio.ogg'
y, sr = librosa.load(audio_data)

hop_seconds = 10
window_seconds = 3
n_mfcc = 20

mfcc=librosa.feature.mfcc(y=y, sr=sr,
                          hop_length=int(hop_seconds*sr),
                          n_fft=int(window_seconds*sr),
                          n_mfcc=n_mfcc)
mfcc_delta=librosa.feature.delta(mfcc)
mfcc_delta2=librosa.feature.delta(mfcc, order=2)
stacked=np.vstack((mfcc, mfcc_delta, mfcc_delta2))
features=stacked.T #librosa возвращает где MFCC идут в ряд, а для модели нужно будет в столбец.

from sklearn.datasets import make_blobs
X, y_true=make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1]);

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4)
gmm.fit(X)
labels=gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');