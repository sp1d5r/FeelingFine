# Feeling Fine?
A project used to recognise emotions from speech - I'm sure you've been in a position where 
someone you care for has hit you with the "I'm fine!", but does that really mean they are fine?

It's the most common lie told...  

The sad truth is if you someone was going through something sad, you'd probably 
try and help out however you can. 

This is my solution. I want to make a Neural Network that can decipher emotions from sound alone.
This is the first leg in my attempt to make a moral neural network companion/therapist. 

As with any of my personal projects, my thought process will be recorded here, and the write up will be 
completed later. 

## Contents 
* [Introduction](https://github.com/sp1d5r/FeelingFine#introduction)
* [Data pre-processing](https://github.com/sp1d5r/FeelingFine#data-pre-processing)
* [Loading Data Set](https://github.com/sp1d5r/FeelingFine#loading-data-set)
* [Initialising the MLP](https://github.com/sp1d5r/FeelingFine#initialising-the-mlp)
* [Training the Model](https://github.com/sp1d5r/FeelingFine#training-the-model)
* [Results](https://github.com/sp1d5r/FeelingFine#results)
* [Saving and Loading the Model](https://github.com/sp1d5r/FeelingFine#saving-and-loading-the-model)
* [Testing custom files](https://github.com/sp1d5r/FeelingFine#testing-custom-files)
* [Citations](https://github.com/sp1d5r/FeelingFine#citations)
* [Social Media](https://github.com/sp1d5r/FeelingFine#social-media)

## Introduction 
So this will be my first solo attempt at a neural network project so a lot of the methods and techniques 
used might be out of date, and the implementation might not follow usual conventions - I'm hoping clean code and 
regular documentation will make up for this. 

As with any neural network problem, we need to find a good data set. We need to find a large set of data 
ideally with emotions - If we cannot find this data set i think another good approach might be to 
look through a bunch of movies, conduct some sentiment analysis on the text, then clip the appropriate sections.
I really really hope this isn't the case. 

Then we need to find an appropriate parameters - With my current knowledge of ANNs we will find different neurons between 
the input and each layer. We will give these neurones different weights, using an activation function it will decide to 
"fire" the value of that neurone in the final value. Obviously we don't do this manually and our ANN will do this itself. 
To be more specific I think I will use an MLP model (forward feeding). 

If I step back to think about how i would try to calculate emotions, I think the best approach for me would include using 
pitch, frequency and tones to recognise emotions. So I think our model will give these neurones more weights. 

Obviously, the task at hand will be a lot more complicated than this and I'm abstracting it quite a bit but that's the 
general direction.

### Dataset 
The dataset I found is the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), this dataset includes 
videos which we wont use. It has 12 actors male and female, using an American Accent. The emotions it recognises is calm, 
happy, sad, angry, fearful, surprise, and disgust expressions

### Analysing Audio
The Librosa library is used for analysing audio files. 


Okay, now we've gone through all of that stuff - 
Let's begin!


## Data pre-processing 
We first need to define some functions to pre process the data. This involves converting the audio file into something the computer can actually understand (numerical values). We are going to use the librosa library and it has some predefined extraction functions. 

We can extract the following information
<img src="https://i.ibb.co/sbmCxfK/Screenshot-2020-11-30-at-13-54-26.png" alt="Table of function" width="600"/>

So, according to my research:
* Chroma : relates to the 12 different pitches, we will be focused with the short term fourier transformation of the sound files. <img src="https://upload.wikimedia.org/wikipedia/commons/2/25/ChromaFeatureCmajorScaleScoreAudioColor.png" alt="(Image of a the 12 different pitches)" width="300"/>
* Melspectogram : This relates to different Mel scale and Spectrogram (Check notebook on more info)
    * Mel scale : The mel scale is the result of non-linear transformations on frequencies to make it easier to plot and record the distance between frequencies
    * Spectrograms : This is the way we plot audio, y axis is hertz, x axis is time, and there is a color spectrum, which ussually represents the decibles. 
* Mel Frequency Cepstral Co-efficients (MFCC) : A feature of sound (similar to edges in photos) / the log of the magnitude of the fourier transformation of sound waves ... 
* Spectral Centroid : The center of mass of the spectrum (also considered the brightness of the sound),
* Spectral Bandwidth : the difference between the max and the min of the spectrum (max change in frequency),
* Spectral Contrast : The differences between the peaks and the valleys in a spectrum, multiple andwidths calculated,
* Roll-Off Frequency : The freqency at which the filter begins to cut off (not sure either)


Okay, now we've gone into what we can extract from the sound waves in a bit more detail I'll briefly explain the thought process behind the selection I will make. I'm deciding to use Chroma since it measures the pitch. I'll use MFCC because it's a feature of sound that the model will be able to use well, I'll also include the spectral centroid, spectral Bandwidth, spectral contrast to try and mimic the variation in frequency based on the idea people have more voice cracks depending on their emotions (although though i am aware this might cause some over fitting in the model). I'll also include the melspectogram and finally I will also include the roll-off frequency as well under the assumption that even if I start the sentence with a lot of energy my emotions determine how fast i speak, the speed of my language determines my frequency (talking slower ussually gives out a lower sound), and the roll-off frequency might help determine this (once again might be over fitting).

```python
import librosa                                             # Audio analyser  
import soundfile                                           # Read the audio files
import os, glob, pickle                                    # Deal with files  
import numpy as np                                         # Numpy used to manipulate dataframes
from sklearn.model_selection import train_test_split       # For testing and training the model 
from sklearn.neural_network import MLPClassifier           # The ANN model  
from sklearn.metrics import accuracy_score                 # used to test the accuracy of our model
```


The function we are going to define takes in a file name, and flags (which parameters to include in extraction), and then returns a data structure which contains the mean of the extracted information. 

Flag names :
* chroma - Chroma Short Term Fourier Transformation (Pitch)
* mfcc - Mel Frequency Cepstral Co-Efficients
* mel - Melspectrogram
* spec_centroid - Spectral Centroid 
* spec_bandwidth - Spectral Bandwidth 
* spec_contrast - Spectral Contrast 
* roll_off - Roll-Off Frequency 

This function goes though each flag and then returns the mean value of it.

```python
'''
Extracting features 
Params : file_name (str), chroma (bool), mfcc (bool), mel (bool), spec_centroid (bool), spec_bandwidth (bool)
           spec_contrast (bool), roll_off (bool) 
'''
def extract_feature(file_name, chroma, mfcc, mel, spec_centroid, spec_bandwidth, spec_contrast, roll_off):
    with soundfile.SoundFile(file_name) as sound_file:
        raw_audio = sound_file.read(dtype="float32") 
        sample_rate = sound_file.samplerate         
        extracted_features = np.array([])
        stft = np.abs(librosa.stft(raw_audio))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            extracted_features = np.hstack((extracted_features, chroma))
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=raw_audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            extracted_features = np.hstack((extracted_features, mfccs))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(raw_audio, sr=sample_rate).T,axis=0)
            extracted_features = np.hstack((extracted_features, mel))
        if spec_centroid:
            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=raw_audio, sr=sample_rate).T,axis=0)
            extracted_features = np.hstack((extracted_features, spec_centroid))
        if spec_bandwidth:
            spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=raw_audio, sr=sample_rate).T,axis=0)
            extracted_features = np.hstack((extracted_features, spec_bandwidth))
        if spec_contrast:
            spec_contrast = np.mean(librosa.feature.spectral_contrast(y=raw_audio, sr=sample_rate).T,axis=0)
            extracted_features = np.hstack((extracted_features, spec_contrast))
        if roll_off:
            roll_off = np.mean(librosa.feature.spectral_rolloff(y=raw_audio, sr=sample_rate).T,axis=0)
            extracted_features = np.hstack((extracted_features, roll_off))
    return extracted_features
```

## Loading Data Set

In this section we will load up the data set and split it into the training and testing set
 ```python
# a dictionary off all emotions we can measure
emotions = {
  '01':'neutral',    # file name XX-XX-01 = neutral 
  '02':'calm',       # file name XX-XX-02 = calm
  '03':'happy',      # file name XX-XX-03 = happy
  '04':'sad',        # file name XX-XX-04 = sad
  '05':'angry',      # file name XX-XX-05 = angry
  '06':'fearful',    # file name XX-XX-06 = fearful
  '07':'disgust',    # file name XX-XX-07 = disgust
  '08':'surprised'   # file name XX-XX-08 = surprised
} 
```
Above we have a dictionary mapping a casted number to an emotion, when going through the data set we are going to load in each entry and then extract it's features. We are then going to split this into training data and test data. We are going to add a parameter for the percentage of data to be in the test data.

```python
# Load the files, extract the features, and split it into the training and test set
def load_data(test_size=0.25):
    x,y=[],[]
    for file in glob.glob("../data/Actor_*/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        feature=extract_feature(file,  chroma=True, mfcc=True, mel=True, spec_centroid=False, spec_bandwidth=False, spec_contrast=False, roll_off=False)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Get the training and testing data
x_train,x_test,y_train,y_test=load_data(0.15)
```

## Initialising the MLP

Here we will initiliase the Mult Level Perception Classifier, with this model there are quite a few parameters we need to consider. So I'm going to go through each one and give a definition of what effect it will have on the model. 

##### hidden_layer_sizes: tuple (length = n_layers - 2, default=(100,) 

This is the number of hidden neurones, it should ideally be between the size of the input layer and the size of the output layer. The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer. The number of hidden neurons should be less than twice the size of the input layer.

##### activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’

This is the activation function, for the hidden layer 
* ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
* ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x))
* ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x)
* ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

##### solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’

This is the solver for the weight optimisation 
* ‘lbfgs’ is an optimizer in the family of quasi-Newton methods
* ‘sgd’ refers to stochastic gradient descent
* ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba 

The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, ‘lbfgs’ can converge faster and perform better.

##### alpha : float, default=0.0001

So we use L2 Regularization, which involve squaring all of the weights and then summing them together using the alpha value (this alpha value is the L2 penalty value). To my understanding a larger alpha value results in more protection against over fitting.

##### batch_size : int, default=’auto’

Size of minibatches for stochastic optimizers. If the solver is ‘lbfgs’, the classifier will not use minibatch. When set to “auto”, batch_size=min(200, n_samples)


##### learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’

Learning rate schedule for weight updates.

* ‘constant’ is a constant learning rate given by ‘learning_rate_init’.
* ‘invscaling’ gradually decreases the learning rate at each time step ‘t’ using an inverse scaling exponent of ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)
* ‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.
Only used when solver='sgd

##### max_iter : int, default=200

Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.

##### epsilon : float, default=1e-8

Value for numerical stability in adam. Only used when solver=’adam’



```python
#Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.005, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), max_iter=550)
```

## Training the Model 

To train the model you use model.fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None). We will use the x training data and the y training data. to get an output,

```python
# Train the model
model.fit(x_train,y_train)
```

Then when you want to test accuracy of the model do the following:
```python
# Calculate the accuracy of our model
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))
```

## Results

Ideally this solution would work and it works relatively well. While you might assume 61% is quite bad, which it is. But the dataset I was working with was not particularly impressive with only 100 pieces of data for each emotion. 

Something I did notice was the spectral information did result in a lot of overfitting for my model - which sucked a litle bit, but it was a good lesson to learn. 


## Saving and loading the model 

Since my model took quite some time to load I am going to use pickle to save the model and use it in other applications - I know this doesn't make a lot of sense for a project with only 60% accuracy but this will be useful for other models i make in the future. 

Saving the model:
```python
# Save the model to a file
filename = 'emotion-model-reloaded.sav'            
pickle.dump(model, open(filename, 'wb'))   
```

Loading the model: 
```python
# Loading the model from a file 
loaded_model = pickle.load(open(filename, 'rb'))

# to recalculate the test 
y_pred=loaded_model.predict(x_test)
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))
```

## Testing custom files

Now that I've exported it into a separate file I can just load up the information from the pickled file and i have a custom function to get the result from the model. I could probably use this in a  

```python
loaded_model = pickle.load(open('emotion-model.sav', 'rb'))

# A file to load a custom audio file 
def load_custom_audio_file(filename):
    x = []
    feature=extract_feature(filename,  chroma=True, mfcc=True, mel=True, spec_centroid=False, spec_bandwidth=False, spec_contrast=False, roll_off=False)
    x.append(feature)
    return x


def predict_for_file(filename):
    return model.predict(load_custom_audio_file(filename))[0]

print(predict_for_file("../data/Actor_01/03-01-03-01-02-02-01.wav"))
```


### Adjustments - Made in hindsight 
* The max_iter when set to 550 gives the best accuracy.
* If alpha is any lower, the accuracy gets reduced.
* If alpha is any higher, the accuracy gets increased.
* Spectral Centroid reduces the accuracy of our model. 
* Spectral Contrast Dramatically reduces the accuracy of our model.
* Spectral bandwidth reduces the accuracy of our model.
* Roll off reduces the accuracy of our model too.




## Citations 
- [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://zenodo.org/record/1188976)
- [Librosa](https://doi.org/10.5281/zenodo.591533)


## Social Media 
- [Linkden - Elijah Ahmad](https://www.linkedin.com/in/elijah-ahmad-658a2b199/)
- [FaceBook - Elijah Ahmad](https://www.facebook.com/elijah.ahmad.71)
- [Instagram - @ElijahAhmad__](https://www.instagram.com/ElijahAhmad__)
- [Snapchat - @Elijah.Ahmad](https://www.snapchat.com/add/elijah.ahmad)
