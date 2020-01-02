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
from keras.optimizers import SGD


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

# Model 0: RNN
model_0 = simple_rnn_model(input_dim=161) # change to 13 if you would like to use MFCC features

"""
As explored in the lesson, we will train the acoustic model with the CTC loss criterion. 

To train your architecture, we will use the train_model function within the train_utils module. 
The train_model function takes three required arguments:

input_to_softmax - a Keras model instance.
pickle_path - the name of the pickle file where the loss history will be saved.
save_model_path - the name of the HDF5 file where the model will be saved.

There are several optional arguments that allow you to have more control over the training process. 

minibatch_size - the size of the minibatches that are generated while training the model (default: 20).
spectrogram - Boolean value dictating whether spectrogram (True) or MFCC (False) features are used for training (default: True).
mfcc_dim - the size of the feature dimension to use when generating MFCC features (default: 13).
optimizer - the Keras optimizer used to train the model (default: SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)).
epochs - the number of epochs to use to train the model (default: 20). If you choose to modify this parameter, make sure that it is at least 20.
verbose - controls the verbosity of the training output in the model.fit_generator method (default: 1).
sort_by_duration - Boolean value dictating whether the training and validation sets are sorted by (increasing) duration before the start of the first epoch (default: False).
The train_model function defaults to using spectrogram features; if you choose to use these features, note that the acoustic model in simple_rnn_model should have input_dim=161. Otherwise, if you choose to use MFCC features, the acoustic model should have input_dim=13.

IMPORTANT NOTE: If we notice that your gradient has exploded in any of the models below, feel free to explore more with gradient clipping (the clipnorm argument in your optimizer) or swap out any SimpleRNN cells for LSTM or GRU cells. 
We can also try restarting the kernel to restart the training process.
"""

train_model(input_to_softmax=model_0, 
            pickle_path='model_0.pickle', 
            save_model_path='model_0.h5',
            minibatch_size=128,
            spectrogram=True) # change to False if you would like to use MFCC features

# Model 1: RNN + TimeDistributed Dense
model_1 = rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                    units=10,
                    activation='relu')

train_model(input_to_softmax=model_1, 
            pickle_path='model_1.pickle', 
            save_model_path='model_1.h5',
            minibatch_size=128,
            spectrogram=True) # change to False if you would like to use MFCC features

# Model 2: CNN + RNN + TimeDistributed Dense
model_2 = cnn_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                        filters=200,
                        kernel_size=11, 
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=200)

train_model(input_to_softmax=model_2, 
            pickle_path='model_2.pickle', 
            save_model_path='model_2.h5',
            minibatch_size=64, 
            spectrogram=True) # change to False if you would like to use MFCC features

# Model 3: Deeper RNN + TimeDistributed Dense
model_3 = deep_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                         units=200,
                         recur_layers=2)

train_model(input_to_softmax=model_3, 
            pickle_path='model_3.pickle', 
            save_model_path='model_3.h5',
            minibatch_size=128,
            spectrogram=True) # change to False if you would like to use MFCC features

# Model 4: Bidirectional RNN + TimeDistributed Dense
model_4 = bidirectional_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                                  units=200)

train_model(input_to_softmax=model_4, 
            pickle_path='model_4.pickle', 
            save_model_path='model_4.h5',
            minibatch_size=128,
            spectrogram=True) # change to False if you would like to use MFCC features

# Conv1D -> RELU -> Batch Normalization -> Bidirectional GRU -> Batch Normalization -> Fully Connected
model_end = final_model(input_dim=161, # change to 13 if you would like to use MFCC features 
                     # CNN parameters
                     cnn_layers = 1,
                     filters=200, kernel_size=11, conv_stride=2, conv_border_mode='valid', dilation=1,
                     cnn_implementation='BN-DR-AC',
                     cnn_dropout=0.3,
                     cnn_activation='relu',
                     # RNN parameters
                     reccur_units=200,  
                     recur_layers=2,
                     recur_type='GRU',
                     recur_implementation=2,
                     reccur_droput=0.3,
                     recurrent_dropout=0.1, 
                     reccur_merge_mode='sum', 
                     # Fully Connected layer parameters
                     fc_units=[200],  # Note: Output layer ofs ize 29 is automatically added in the model
                     fc_dropout=0.3,
                     fc_activation='relu')

train_model(input_to_softmax=model_end,
            epochs=30, 
            minibatch_size=128, ##### Adjust batch size
            optimizer=SGD(lr=0.025, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=4.0),
            sort_by_duration=True,
            pickle_path='model_5.pickle', 
            save_model_path='model_5.h5',
            train_json='train_corpus.json',
            spectrogram=True) # change to False if you would like to use MFCC features

# Variation over model : compare GRU vs LSTM for the recurrent layers
model_end = final_model(input_dim=161, # change to 13 if you would like to use MFCC features 
                     # CNN parameters
                     cnn_layers = 1,
                     filters=200, kernel_size=11, conv_stride=2, conv_border_mode='valid', dilation=1,
                     cnn_implementation='BN-DR-AC',
                     cnn_dropout=0.3,
                     cnn_activation='relu',
                     # RNN parameters
                     reccur_units=200,  
                     recur_layers=2,
                     recur_type='LSTM',
                     recur_implementation=2,
                     reccur_droput=0.3,
                     recurrent_dropout=0.1, 
                     reccur_merge_mode='sum', 
                     # Fully Connected layer parameters
                     fc_units=[200],  # Note: Output layer ofs ize 29 is automatically added in the model
                     fc_dropout=0.3,
                     
train_model(input_to_softmax=model_end,
            epochs=30, 
            minibatch_size=128, ##### Adjust batch size
            optimizer=SGD(lr=0.025, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=4.0),
            sort_by_duration=True,
            pickle_path='model_6.pickle', 
            save_model_path='model_6.h5',
            train_json='train_corpus.json',
            spectrogram=True) # change to False if you would like to use MFCC features

# Variation over model : compare 'concat' versus 'sum' in the merge function of the Bidirectional RNN layers
model_end = final_model(input_dim=161, # change to 13 if you would like to use MFCC features 
                     # CNN parameters
                     cnn_layers = 1,
                     filters=200, kernel_size=11, conv_stride=2, conv_border_mode='valid', dilation=1,
                     cnn_implementation='BN-DR-AC',
                     cnn_dropout=0.3,
                     cnn_activation='relu',
                     # RNN parameters
                     reccur_units=200,  
                     recur_layers=2,
                     recur_type='GRU',
                     recur_implementation=2,
                     reccur_droput=0.3,
                     recurrent_dropout=0.1, 
                     reccur_merge_mode='concat', 
                     # Fully Connected layer parameters
                     fc_units=[200],  # Note: Output layer ofs ize 29 is automatically added in the model
                     fc_dropout=0.3,
                     fc_activation='relu')

train_model(input_to_softmax=model_end,
            epochs=30, 
            minibatch_size=128, ##### Adjust batch size
            optimizer=SGD(lr=0.025, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=4.0),
            sort_by_duration=True,
            pickle_path='model_7.pickle', 
            save_model_path='model_7.h5',
            train_json='train_corpus.json',
            spectrogram=True) # change to False if you would like to use MFCC features

# Variation over model : compare the impact on using a Spectrogram or MFCC input format
model_end = final_model(input_dim=13, 
                     # CNN parameters
                     cnn_layers = 1,
                     filters=200, kernel_size=11, conv_stride=2, conv_border_mode='valid', dilation=1,
                     cnn_implementation='BN-DR-AC',
                     cnn_dropout=0.3,
                     cnn_activation='relu',
                     # RNN parameters
                     reccur_units=200,  
                     recur_layers=2,
                     recur_type='GRU',
                     recur_implementation=2,
                     reccur_droput=0.3,
                     recurrent_dropout=0.1, 
                     reccur_merge_mode='sum', 
                     # Fully Connected layer parameters
                     fc_units=[200],  # Note: Output layer ofs ize 29 is automatically added in the model
                     fc_dropout=0.3,
                     fc_activation='relu')

train_model(input_to_softmax=model_end,
            epochs=30, 
            minibatch_size=128, ##### Adjust batch size
            optimizer=SGD(lr=0.025, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=4.0),
            sort_by_duration=True,
            pickle_path='model_8.pickle', 
            save_model_path='model_8.h5',
            train_json='train_corpus.json',
            spectrogram=False)

# Variation over model : compare the impact of using a padding 'same' instade of 'valid' in the convolution layer
model_end = final_model(input_dim=161, # change to 13 if you would like to use MFCC features 
                     # CNN parameters
                     cnn_layers = 1,
                     filters=200, kernel_size=11, conv_stride=2, conv_border_mode='same', dilation=1,
                     cnn_implementation='BN-DR-AC',
                     cnn_dropout=0.3,
                     cnn_activation='relu',
                     # RNN parameters
                     reccur_units=200,  
                     recur_layers=2,
                     recur_type='GRU',
                     recur_implementation=2,
                     reccur_droput=0.3,
                     recurrent_dropout=0.1, 
                     reccur_merge_mode='sum', 
                     # Fully Connected layer parameters
                     fc_units=[200],  # Note: Output layer ofs ize 29 is automatically added in the model
                     fc_dropout=0.3,
                     fc_activation='relu')

train_model(input_to_softmax=model_end,
            epochs=30, 
            minibatch_size=128, ##### Adjust batch size
            optimizer=SGD(lr=0.025, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=4.0),
            sort_by_duration=True,
            pickle_path='model_9.pickle', 
            save_model_path='model_9.h5',
            train_json='train_corpus.json',
            spectrogram=True) # change to False if you would like to use MFCC features

# Variation over model : compare the impact of using a dilation of 2 in the convolution layer (padding 'same'+stride=1)
model_end = final_model(input_dim=161, # change to 13 if you would like to use MFCC features 
                     # CNN parameters
                     cnn_layers = 1,
                     filters=200, kernel_size=11, conv_stride=1, conv_border_mode='same', dilation=2,
                     cnn_implementation='BN-DR-AC',
                     cnn_dropout=0.3,
                     cnn_activation='relu',
                     # RNN parameters
                     reccur_units=200,  
                     recur_layers=2,
                     recur_type='GRU',
                     recur_implementation=2,
                     reccur_droput=0.3,
                     recurrent_dropout=0.1, 
                     reccur_merge_mode='sum', 
                     # Fully Connected layer parameters
                     fc_units=[200],  # Note: Output layer ofs ize 29 is automatically added in the model
                     fc_dropout=0.3,
                     fc_activation='relu')

train_model(input_to_softmax=model_end,
            epochs=30, 
            minibatch_size=128, ##### Adjust batch size
            optimizer=SGD(lr=0.025, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=4.0),
            sort_by_duration=True,
            pickle_path='model_10.pickle', 
            save_model_path='model_10.h5',
            train_json='train_corpus.json',
            spectrogram=True) # change to False if you would like to use MFCC features

 