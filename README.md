#Feeling Fine?
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
* Introduction 
* Citations
* Social Media

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


## Citations 
- [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://zenodo.org/record/1188976)
- [Librosa](https://doi.org/10.5281/zenodo.591533)


## Social Media 
- [Linkden - Elijah Ahmad](https://www.linkedin.com/in/elijah-ahmad-658a2b199/)
- [FaceBook - Elijah Ahmad](https://www.facebook.com/elijah.ahmad.71)
- [Instagram - @ElijahAhmad__](https://www.instagram.com/ElijahAhmad__)
- [Snapchat - @Elijah.Ahmad](https://www.snapchat.com/add/elijah.ahmad)