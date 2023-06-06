import collections
import contextlib
import copy
import sys
import wave

from IPython.core.display_functions import clear_output
from sklearn import preprocessing
from sklearn.cluster import SpectralClustering
from webrtcvad_wrapper import VAD
import numpy as np
import os
import librosa.display
from pydub import AudioSegment, audio_segment
from pydub.silence import split_on_silence
import soundfile as sf   #   pip install pysoundfile
from pathlib import Path
import webrtcvad
import pickle
from sklearn.mixture import GaussianMixture

audiofile = "input/Vacation2P.ogg"

exportpath = "export/frames/"
name = Path(audiofile).stem
print(name)
os.path.join(name + '.wav')
namewav = os.path.join(name + '.wav')
print(namewav)

#convert to wav
data, samplerate = sf.read(audiofile)
sf.write("input/" + namewav, data, samplerate)

wavfile = os.path.join("input/" + namewav)
print(wavfile)

#clean fog and silense
sound_file = AudioSegment.from_wav(wavfile)
audio_chunks = split_on_silence(sound_file, min_silence_len=1000, silence_thresh=-40)

# #get frames
# for i, chunk in enumerate(audio_chunks):
#     out_file = exportpath + "chunk{0}.wav".format(i)
#     print("exporting", out_file)
#     chunk.export(out_file, format="wav")

vad = webrtcvad.Vad()

# mfcclist = dict()
# for ch in os.listdir(exportpath):
#     file = ch
#     y,sr = librosa.load(exportpath + file, sr=sr)


sr=16000
n_mfcc = 512
hop_length = 45
win_length = 10
# y,sr = librosa.load(wavfile,sr=sr)
# mfcc = librosa.feature.mfcc(y=y, sr=sr,
#                             hop_length=hop_length,
#                             n_fft=int(win_length * sr),
#                             dct_type=3,
#                             n_mfcc=n_mfcc)
# mfcc_delta = librosa.feature.delta(mfcc)
# mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
# stacked = np.vstack((mfcc, mfcc_delta, mfcc_delta2))
# features = stacked.T  # librosa возвращает где MFCC идут в ряд, а для модели нужно будет в столбец.
# #print(features.shape())
# print(features.min())
# print(features.max())

# возьмём сегменты от chunk-000 до chunk-100
SR = 8000 # sample rate
N_MFCC = 13 # number of MFCC to extract
N_FFT = 0.032  # length of the FFT window in seconds
HOP_LENGTH = 0.010
N_COMPONENTS = 16 # number of gaussians
COVARINACE_TYPE = 'full' # cov type for GMM

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
# def extract_features(y_, sr_, window_seconds, hop_seconds, n_mfcc):
#     mfcc=librosa.feature.mfcc(y=y_, sr=sr_,
#                               hop_length=int(hop_seconds*sr),
#                               n_fft=int(window_seconds*sr),
#                               n_mfcc=n_mfcc)
#     mfcc_delta=librosa.feature.delta(mfcc)
#     mfcc_delta2=librosa.feature.delta(mfcc, order=2)
#     stacked=np.vstack((mfcc, mfcc_delta, mfcc_delta2))
#     features=stacked.T #librosa возвращает где MFCC идут в ряд, а для модели нужно будет в столбец.
#     return features
def extract_features(y, sr, window, hop, n_mfcc):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=int(hop*sr), n_fft=int(window*sr), n_mfcc=n_mfcc, dct_type=2)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.vstack((mfcc, mfcc_delta, mfcc_delta2))
    return stacked.T


def map_adaptation(gmm, data, max_iterations=300, likelihood_threshold=1e-20, relevance_factor=16):
    N = data.shape[0]
    D = data.shape[1]
    K = gmm.n_components

    mu_new = np.zeros((K, D))
    n_k = np.zeros((K, 1))

    mu_k = gmm.means_
    cov_k = gmm.covariances_
    pi_k = gmm.weights_

    old_likelihood = gmm.score(data)
    new_likelihood = 0
    iterations = 0
    while (abs(old_likelihood - new_likelihood) > likelihood_threshold and iterations < max_iterations):
        iterations += 1
        old_likelihood = new_likelihood
        z_n_k = gmm.predict_proba(data)
        n_k = np.sum(z_n_k, axis=0)

        for i in range(K):
            temp = np.zeros((1, D))
            for n in range(N):
                temp += z_n_k[n][i] * data[n, :]
            mu_new[i] = (1 / n_k[i]) * temp

        adaptation_coefficient = n_k / (n_k + relevance_factor)
        for k in range(K):
            mu_k[k] = (adaptation_coefficient[k] * mu_new[k]) + ((1 - adaptation_coefficient[k]) * mu_k[k])
        gmm.means_ = mu_k

        log_likelihood = gmm.score(data)
        new_likelihood = log_likelihood
        print(log_likelihood)
    return gmm

def rearrange(labels, n):
    seen = set()
    distinct = [x for x in labels if x not in seen and not seen.add(x)]
    correct = [i for i in range(n)]
    dict_ = dict(zip(distinct, correct))
    return [x if x not in dict_ else dict_[x] for x in labels]

#читаем сигнал
y, sr = librosa.load(wavfile, sr=sr)
#первым шагом делаем pre-emphasis: усиление высоких частот
pre_emphasis = 0.97
y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

#все что ниже фактически взято с гитхаба webrtcvad с небольшими изменениями
#vad = VAD(2) # агрессивность VAD
audio = np.int16(y/np.max(np.abs(y)) * 32767)

frames = frame_generator(10, audio, sr)
frames = list(frames)
segments = vad_collector(sr, 50, 200, vad, frames)

if not os.path.exists('data/chunks'): os.makedirs('data/chunks')
for i, segment in enumerate(segments):
    chunk_name = 'data/chunks/chunk-%003d.wav' % (i,)
    # vad добавляет в конце небольшой кусочек тишины, который нам не нужен
    write_wave(chunk_name, segment[0: len(segment)-int(100*sr/1000)], sr)

# extract MFCC, first and second derivatives
FEATURES_FROM_FILE = True

feature_file_name = 'data/features_{0}.pkl'.format(N_MFCC)

if FEATURES_FROM_FILE:
    ubm_features=pickle.load(open(feature_file_name, 'rb'))
else:
    ubm_features = extract_features(np.array(y), sr, window=N_FFT, hop=HOP_LENGTH, n_mfcc=N_MFCC)
    ubm_features = preprocessing.scale(ubm_features)
    pickle.dump(ubm_features, open(feature_file_name, "wb"))

# UBM Train
UBM_FROM_FILE = True

ubm_file_name = 'data/ubm_{0}_{1}_{2}MFCC.pkl'.format(N_COMPONENTS, COVARINACE_TYPE, N_MFCC)

if UBM_FROM_FILE:
    ubm = pickle.load(open(ubm_file_name, 'rb'))
else:
    ubm = GaussianMixture(n_components=N_COMPONENTS, covariance_type=COVARINACE_TYPE)
    ubm.fit(ubm_features)
    pickle.dump(ubm, open(ubm_file_name, "wb"))

print(ubm.score(ubm_features))

SV = []

for i in range(101):
    clear_output(wait=True)
    fname = 'data/chunks/chunk-%003d.wav' % (i,)
    print('UBM MAP adaptation for {0}'.format(fname))
    y_, sr_ = librosa.load(fname, sr=None)
    f_ = extract_features(y_, sr_, window=N_FFT, hop=HOP_LENGTH, n_mfcc=N_MFCC)
    f_ = preprocessing.scale(f_)
    gmm = copy.deepcopy(ubm)
    gmm = map_adaptation(gmm, f_, max_iterations=1, relevance_factor=16)
    sv = gmm.means_.flatten()  # получаем супервектор мю
    sv = preprocessing.scale(sv)
    SV.append(sv)
SV = np.array(SV)
clear_output()
print(SV.shape)

N_CLUSTERS = 2
sc = SpectralClustering(n_clusters=N_CLUSTERS, affinity='cosine')
labels = sc.fit_predict(SV) # кластеры могут быть не упорядочены, напр. [2,1,1,0,2]
labels = rearrange(labels, N_CLUSTERS) # эта функция упорядочивает кластеры [0,1,1,2,0]
print(labels)
# глядя на результат, понимаем, что 1 - это интервьюер. выведем все номера сегментов
print([i for i, x in enumerate(labels) if x == 1])
