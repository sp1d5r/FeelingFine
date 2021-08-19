from time import sleep
from flask import Flask, flash, request, redirect, url_for, render_template, abort
# import librosa                                             # Audio analyser
# import soundfile                                           # Read the audio files
import pickle                                               # Deal with files
import numpy as np                                         # Numpy used to manipulate dataframes


app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.wav', '.mp3', '.aac', '.flac']
app.config['UPLOAD_PATH'] = 'uploads'

# Loading the Model
# loaded_model = pickle.load(open('emotion-model.sav', 'rb'))


def extract_feature(file_name, chroma, mfcc, mel, spec_centroid, spec_bandwidth, spec_contrast, roll_off):
    # with soundfile.SoundFile(file_name) as sound_file:
    #     raw_audio = sound_file.read(dtype="float32")
    #     sample_rate = sound_file.samplerate
    #     extracted_features = np.array([])
    #     stft = np.abs(librosa.stft(raw_audio))
    #     if chroma:
    #         chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    #         extracted_features = np.hstack((extracted_features, chroma))
    #     if mfcc:
    #         mfccs=np.mean(librosa.feature.mfcc(y=raw_audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    #         extracted_features = np.hstack((extracted_features, mfccs))
    #     if mel:
    #         mel = np.mean(librosa.feature.melspectrogram(raw_audio, sr=sample_rate).T,axis=0)
    #         extracted_features = np.hstack((extracted_features, mel))
    #     if spec_centroid:
    #         spec_centroid = np.mean(librosa.feature.spectral_centroid(y=raw_audio, sr=sample_rate).T,axis=0)
    #         extracted_features = np.hstack((extracted_features, spec_centroid))
    #     if spec_bandwidth:
    #         spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=raw_audio, sr=sample_rate).T,axis=0)
    #         extracted_features = np.hstack((extracted_features, spec_bandwidth))
    #     if spec_contrast:
    #         spec_contrast = np.mean(librosa.feature.spectral_contrast(y=raw_audio, sr=sample_rate).T,axis=0)
    #         extracted_features = np.hstack((extracted_features, spec_contrast))
    #     if roll_off:
    #         roll_off = np.mean(librosa.feature.spectral_rolloff(y=raw_audio, sr=sample_rate).T,axis=0)
    #         extracted_features = np.hstack((extracted_features, roll_off))
    # return extracted_features
    return None


def load_custom_audio_file(filename):
    x = []
    feature=extract_feature(filename,  chroma=True, mfcc=True, mel=True, spec_centroid=False, spec_bandwidth=False, spec_contrast=False, roll_off=False)
    x.append(feature)
    return x


def predict_for_file(filename):
    return loaded_model.predict(load_custom_audio_file(filename))[0]


SubmitButtonColor = {
  "Submit" : "#F5DCE0",
  'neutral': "#EAEAEA",
  'calm': '#FFFFB5',
  'happy': '#CCE2CB',
  'sad': '#FFAEA5',
  'angry': '#FF968A',
  'fearful': '#CBAACB',
  'disgust': '#97C1A9',
  'surprised': '#FFC8A2'
}


@app.route('/')
def ind():
    return render_template('index.html',
                           result=request.args.get('result', '#F5DCE0'),
                           value=request.args.get('value', 'Submit'),
                           filename=request.args.get('filename', 'Choose File'))


@app.route('/', methods=['GET', 'POST'])
def index():
    result = "Submit"
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
        sleep(5)
        result = predict_for_file(uploaded_file.filename)
        sleep(2)
        return redirect(url_for('ind', result=SubmitButtonColor[result], value=result, filename=uploaded_file.filename))
    return render_template('index.html', result=SubmitButtonColor[result], value=result, filename="Choose File")
