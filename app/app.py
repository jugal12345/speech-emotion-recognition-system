from flask import Flask, render_template, request
from joblib import load
import librosa
import soundfile
import os
import numpy as np

app=Flask(__name__)


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128))
            result=np.hstack((result, mel))
    return result







@app.route('/')
def hello_world():
    model=load('model.joblib')
    return render_template('index.html')


@app.route("/upload", methods=["POST"])
def upload():
    if "audio-file" not in request.files:
        return "no"
    file = request.files["audio-file"]
    if file.filename == "":
        return "no"
    if file and allowed_file(file.filename):
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        model=load('model.joblib')
        pred=model.predict([feature])
        return str(pred[0])
    else:
        return "no"
    
    
def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in {"mp3", "wav", "ogg"}