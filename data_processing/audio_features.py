import numpy
import numpy as np
import scipy.io.wavfile as wav
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import os
import math
import json
import tensorflow_hub as hub

def extract_sentences(file, script_dir = "../data/GENEA/source/script/"):
    f = open(script_dir + file + ".json")
    script = json.load(f)
    f.close()
    script = script[0]["alternatives"][0]["words"]

    sentences = []

    sentence = ""
    start = -1
    for word in script:
        if word["word"][-1] == ".":
            sentence += word["word"]
            sentences.append({
                "start_time": start,
                "end_time": word["end_time"],
                "sentence": sentence
            })
            sentence = ""
            start = -1
        else:
            if start == -1:
                start = word["start_time"]
            sentence += word["word"] + " "

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    arr = []
    for el in sentences:
        arr.append(el["sentence"])

    embeddings = embed(arr)

    for i in range(len(embeddings)):
        res = embeddings[i].numpy()
        sentences[i]["embeddings"] = res

    return sentences

def extract_melspec(audio_dir, files, destpath, fps, sentence_path):
    for f in range(len(files)):
        sentences = extract_sentences(sentence_path[f])
        file = os.path.join(audio_dir, files[f] + '.wav')
        outfile = destpath + '/' + files[f] + '.npy'
        
        print('{}\t->\t{}'.format(file,outfile))
        fs,X = wav.read(file)
        X = X.astype(float)/math.pow(2,15)

        hop_len=int(fs/fps)

        melspec_feat = np.empty((539, 0))
        start = 0
        for i in range(len(sentences)):
            end = int(float(sentences[i]["end_time"][:-1]) * fs) if i < len(sentences) - 1 else len(X)

            features = librosa.feature.melspectrogram(y=X[start:end], sr=fs, n_fft=2048, hop_length=hop_len, n_mels=27, fmin=0.0, fmax=8000)
            features = np.log(features)
            embeddings = np.tile(sentences[i]["embeddings"], (features.shape[1], 1))
            embeddings = np.transpose(embeddings)
            features = np.append(features, embeddings, axis=0)

            melspec_feat = np.append(melspec_feat, features, axis = 1)

        np.save(outfile,np.transpose(melspec_feat))
        return melspec_feat

def get_dfd(mfcc_features):
    mfcc_features_1d = mfcc_features[2:] - mfcc_features[1:-1]
    mfcc_features_2d = mfcc_features_1d[1:] - mfcc_features_1d[:-1]
    return np.concatenate((mfcc_features, mfcc_features_1d, mfcc_features_2d), axis=0)

def extract_mfcc(audio_dir, files, destpath, sentence_path):


    for f in range(len(files)):
        sentences = extract_sentences(sentence_path[f])
        file = os.path.join(audio_dir, files[f] + ".wav")
        outfile = destpath + '/' + files[f] + '.npy'

        print('{}\t->\t{}'.format(file, outfile))
        X, sr = librosa.load(file)

        mfcc_features = np.empty((549,0))
        start = 0
        for i in range(len(sentences)):
            end = int(float(sentences[i]["end_time"][:-1]) * sr) if i < len(sentences) - 1 else len(X)

            features = get_dfd(librosa.feature.mfcc(X[start:end], sr=sr, n_mfcc=14) / 1000)
            embeddings = np.tile(sentences[i]["embeddings"], (features.shape[1], 1))
            embeddings = np.transpose(embeddings)
            features = np.append(features, embeddings, axis=0)

            mfcc_features = np.append(mfcc_features, features, axis=1)

            start = end

        np.save(outfile, np.transpose(mfcc_features))
        return mfcc_features

if __name__ == "__main__":
    audio_dir = "../data"
    files = ["TestSeq001-2"]
    destpath = "../data"
    fps = 20

    extract_mfcc(audio_dir, files, destpath, ["../data/TestSeq002"])

    extract_melspec(audio_dir, files, destpath, fps, ["../data/TestSeq002"])







