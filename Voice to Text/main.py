# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 23:39:48 2020

@author: Tanmay Thakur
"""

# Check Env

# Check GPU
from tensorflow.python.client import device_lib
from tensorflow import test

if test.is_gpu_available:
    devices=device_lib.list_local_devices()
    gpu=[(x.physical_device_desc) for x in devices if x.device_type == 'GPU']
    print("GPU :", gpu)

# Check framework versions being used 
import tensorflow as tf
import keras as K
import sys
print("Python", sys.version)
print("Tensorflow version", tf.__version__)
print("Keras version", K.__version__)

# import NN architectures for speech recognition
from sample_models import *
# import function for training acoustic model
from train_utils import train_model

# Import Additional Libraries
from IPython.display import Markdown, display, Audio
from data_generator import vis_train_features, plot_raw_audio, plot_spectrogram_feature, plot_mfcc_feature
from keras.backend.tensorflow_backend import set_session


# Extract label and audio features for a single training example
vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path = vis_train_features()

# plot audio signal
plot_raw_audio(vis_raw_audio)
# print length of audio signal
display(Markdown('**Shape of Audio Signal** : ' + str(vis_raw_audio.shape)))
# print transcript corresponding to audio clip
display(Markdown('**Transcript** : ' + str(vis_text)))
# play the audio file
Audio(vis_audio_path)

"""
Acoustic Features for Speech Recognition
We won't use the raw audio waveform as input to your model. 
Code first performs a pre-processing step to convert the raw audio to a feature representation that has historically proven successful for ASR models. 
Acoustic model will accept the feature representation as input.

We will explore two possible feature representations. 

Spectrograms
The first option for an audio feature representation is the spectrogram. 

The code returns the spectrogram as a 2D tensor, where the first (vertical) dimension indexes time, and the second (horizontal) dimension indexes frequency. 
To speed the convergence of the algorithm, we have also normalized the spectrogram. (We can see this quickly in the visualization below by noting that the mean value hovers around zero, and most entries in the tensor assume values close to zero.)
"""

# plot normalized spectrogram
plot_spectrogram_feature(vis_spectrogram_feature)
# print shape of spectrogram
display(Markdown('**Shape of Spectrogram** : ' + str(vis_spectrogram_feature.shape)))

"""
Mel-Frequency Cepstral Coefficients (MFCCs)
The second option for an audio feature representation is MFCCs. 
Just as with the spectrogram features, the MFCCs are normalized in the supplied code.

The main idea behind MFCC features is the same as spectrogram features: at each time window, the MFCC feature yields a feature vector that characterizes the sound within the window. 
Note that the MFCC feature is much lower-dimensional than the spectrogram feature, which could help an acoustic model to avoid overfitting to the training dataset.
"""

# plot normalized MFCC
plot_mfcc_feature(vis_mfcc_feature)
# print shape of MFCC
display(Markdown('**Shape of MFCC** : ' + str(vis_mfcc_feature.shape)))

# allocate 50% of GPU memory
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

model_0 = simple_rnn_model(input_dim=161) # change to 13 if you would like to use MFCC features

"""

"""