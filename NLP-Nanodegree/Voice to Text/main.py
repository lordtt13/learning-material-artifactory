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
from glob import glob
import numpy as np
import _pickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter, defaultdict
from nltk import trigrams
from nltk.corpus import brown

from data_generator import AudioGenerator, plot_raw_audio, plot_spectrogram_feature
from utils import int_sequence_to_text

from IPython.display import Markdown, display, Audio
from data_generator import vis_train_features, plot_mfcc_feature
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import SGD

sns.set_style(style='white')
nltk.download('brown')


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

# Variation over model : compare the impact of using a dilation of 2 with 2 convolution layers (padding 'same'+stride=1)
model_end = final_model(input_dim=161, # change to 13 if you would like to use MFCC features 
                     # CNN parameters
                     cnn_layers = 2,
                     filters=200, kernel_size=11, conv_stride=1, conv_border_mode='same', dilation=2,
                     cnn_implementation='BN-DR-AC',
                     cnn_dropout=0.3,
                     cnn_activation='relu',
                     # RNN parameters=
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
            epochs=30, =
            minibatch_size=128, ##### Adjust batch size
            optimizer=SGD(lr=0.025, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=4.0),
            sort_by_duration=True,
            pickle_path='model_11.pickle', 
            save_model_path='model_11.h5',
            train_json='train_corpus.json',
            spectrogram=True) # change to False if you would like to use MFCC features 

# Variation over model : compare the impact of using a dilation of 4 with 2 convolution layers (padding 'same'+stride=1)
model_end = final_model(input_dim=161, # change to 13 if you would like to use MFCC features 
                     # CNN parameters
                     cnn_layers = 2,
                     filters=200, kernel_size=11, conv_stride=1, conv_border_mode='same', dilation=2,
                     cnn_implementation='BN-DR-AC',
                     cnn_dropout=0.3,
                     cnn_activation='relu',
                     # RNN parameters=
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
            pickle_path='model_12.pickle', 
            save_model_path='model_12.h5',
            train_json='train_corpus.json',
            spectrogram=True) # change to False if you would like to use MFCC features

# Final model trained on 30 epochs and development training set
model_end = final_model(input_dim=161, # change to 13 if you would like to use MFCC features
                        # CNN parameters
                        cnn_layers =3,
                        filters=350, kernel_size=11, conv_stride=1, conv_border_mode='same', dilation=4,
                        cnn_implementation='BN-DR-AC',
                        cnn_dropout=0.3,
                        cnn_activation='relu',
                        # RNN parameters
                        reccur_units=200,  #29,
                        recur_layers=3,
                        recur_type='LSTM',
                        recur_implementation=2,
                        reccur_droput=0.3,
                        recurrent_dropout=0.1, 
                        reccur_merge_mode='sum', 
                        # Fully Connected layer parameters
                        fc_units=[400, 200, 100],
                        fc_dropout=0.3,
                        fc_activation='relu')

train_model(input_to_softmax=model_end,
            epochs=30, 
            minibatch_size=128, ##### Adjust batch size
            optimizer=SGD(lr=0.025, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=4.0),
            sort_by_duration=True,
            pickle_path='model_candidate.pickle', 
            save_model_path='model_candidate.h5',
            train_json='train_corpus.json',
            spectrogram=True) # change to False if you would like to use MFCC features

# obtain the paths for the saved model history
all_pickles = sorted(glob("results/*.pickle"))
# extract the name of each model
model_names = [item[8:-7] for item in all_pickles]
# extract the loss history for each model
valid_loss = [pickle.load( open( i, "rb" ) )['val_loss'] for i in all_pickles]
train_loss = [pickle.load( open( i, "rb" ) )['loss'] for i in all_pickles]
# save the number of epochs used to train each model
num_epochs = [len(valid_loss[i]) for i in range(len(valid_loss))]

fig = plt.figure(figsize=(16,5))

# plot the training loss vs. epoch for each model
ax1 = fig.add_subplot(121)
for i in range(len(all_pickles)):
    ax1.plot(np.linspace(1, num_epochs[i], num_epochs[i]), 
            train_loss[i], label=model_names[i])
# clean up the plot
ax1.legend()  
ax1.set_xlim([1, max(num_epochs)])
plt.xlabel('Epoch')
plt.ylabel('Training Loss')

# plot the validation loss vs. epoch for each model
ax2 = fig.add_subplot(122)
for i in range(len(all_pickles)):
    ax2.plot(np.linspace(1, num_epochs[i], num_epochs[i]), 
            valid_loss[i], label=model_names[i])
# clean up the plot
ax2.legend()  
ax2.set_xlim([1, max(num_epochs)])
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.show()

# Obtain Predictions

def get_predictions(index, partition, input_to_softmax, model_path):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_validation_data()
    
    # obtain the true transcription and the audio features 
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')
        
    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])] 
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length)[0][0])+1).flatten().tolist()
    
    # play the audio file, and display the true and predicted transcriptions
    print('-'*80)
    Audio(audio_path)
    print('True transcription:\n' + '\n' + transcr)
    print('-'*80)
    print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
    print('-'*80)
    
get_predictions(index=0, 
                partition='train',
                #input_to_softmax=final_model(input_dim=161, # change to 13 if you would like to use MFCC features
                #                             filters=250, kernel_size=11, 
                #                             conv_stride=2, conv_border_mode='valid',
                #                             units=250, recur_layers=3), 
                
                input_to_softmax=model_end,
                model_path='./results/model_end.h5')

get_predictions(index=0, 
                partition='validation',
                #input_to_softmax=final_model(input_dim=161, # change to 13 if you would like to use MFCC features
                #                             filters=200, kernel_size=11, 
                #                             conv_stride=2, conv_border_mode='valid',
                #                             units=250, recur_layers=3), 
                input_to_softmax=model_end,
                model_path='./results/model_end.h5')

# Spell corrector based on the training corpus
def words(text): return re.findall(r'\w+', text.lower())

# Prepare a corpus from the training data
# (using helpers provided in data_generator.py) 
corpus = AudioGenerator()
corpus.load_train_data(desc_file='train-360_corpus.json')
corpus_text = corpus.train_texts

print("Corpus_text length: ", len(corpus_text), "\nFirst example : ", corpus_text[0])

# Create a count the words in the corpus

# For each sentence in the corpus 
words_list = [ word for sentence in corpus_text for word in sentence.lower().split() ]

print("words_list length: ", len(words_list), "\nFirst 50 examples: ", words_list[:50])

WORDS=Counter(words_list)

print("\nMost comon words:", WORDS.most_common(20))

# Spell checker by Peter Norvig

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# Adjust get_predictions() method to return the True and predicted transcriptions

def do_predictions(index, partition, input_to_softmax, model_path):
    """ Return the True and predicted transcriptions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_validation_data()
    
    # obtain the true transcription and the audio features 
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')
        
    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])] 
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length)[0][0])+1).flatten().tolist()
    
    # Return the true and predicted transcriptions
    Audio(audio_path)
    return transcr, ''.join(int_sequence_to_text(pred_ints))

# Retrieve a prediction
label, prediction = do_predictions(index=0,
                                   partition='validation',  # or train
                                   input_to_softmax=model_end,
                                   model_path='./results/model_end.h5')

# Spell Correction
corrected_prediction=[]
for word in prediction.lower().split():
    corrected_prediction.append(correction(word))
    
print("True transcription:         ", label) 
print("Model raw prediction:       ", prediction)
print("Spell corrected prediction: ", ' '.join(corrected_prediction))

# Rebuild the words list
words_list = brown.words(categories='adventure') + brown.words(categories='romance') + brown.words(categories='fiction')

print("words_list length: ", len(words_list), "\nFirst 50 examples: ", words_list[:50])

WORDS=Counter(words_list)

print("\nMost comon words:", WORDS.most_common(20))

# Spell Correction
corrected_prediction=[]
for word in prediction.lower().split():
    corrected_prediction.append(correction(word))
    
print("True transcription:         ", label) 
print("Model raw prediction:       ", prediction)
print("Spell corrected prediction: ", ' '.join(corrected_prediction))

# Create a trigram language model, base on our training set corpus
language_model = defaultdict(lambda: defaultdict(lambda: 0))
 
for sentence in corpus_text:
    for w1, w2, w3 in trigrams(sentence.lower().split(), pad_right=True, pad_left=True):
        language_model[(w1, w2)][w3] += 1

        
# Transform the counts intoto probabilities
for w1_w2 in language_model:
    total_count = float(sum(language_model[w1_w2].values()))
    for w3 in language_model[w1_w2]:
        language_model[w1_w2][w3] /= total_count
        
        
next_words = language_model["the", "two"]
print('Debug : word candidates for ["the", "two"] sequence : ', next_words)

corrected_prediction=[]

predicted_words = prediction.lower().split()

# Comment this code to avoid debug mode/unit test (need a prediction with more correct words, especially for the 2 first words...)
print("\n\nDeeug : unit test" )
label = "also a popular contrivance whereby love making may be suspended but not stopped during the picnic season"
predicted_words=["also", "a", "popela", "contrivance", "whereby", "lof", "making", "may", "beses", "suspended", "but", "not", "stok", "during", "the", "picnic", "season"]

print("Predicted_words=", predicted_words) 

word1=""
word2=""

for i in range(len(predicted_words)):
    # We assume the first word is correct
    if i==0:
        word1=predicted_words[i]
        corrected_prediction.append(word1)
    elif i==1:
        # We also assume the second word is corrcet (TODO: check with a bigram model)
        word2=predicted_words[i]
        corrected_prediction.append(word2)
    else :
        new_word2=predicted_words[i]
        # if current words is not in the corpus, check if a trigram prediction exist
        if predicted_words[i] not in words_list and len(language_model[word1, word2].items())>0 and max(language_model[word1, word2].items(), key=lambda x: x[1])[0] is not None:
            
            # Replace with the word which has the highest probability to follows the previous 2 words
            # TODO : check if test condition is ok + how to eventually break tie ?
            corrected_word = max(language_model[word1, word2].items(), key=lambda x: x[1])
            corrected_prediction.append(corrected_word[0])
            #print("Debug : corrected_word: ", corrected_word)
            print("correcting: ", predicted_words[i], "-> ", corrected_word[0])
            new_word2=corrected_word[0]
        else:
            # else Keep the current predicted words
            # TODO : maybe also compare if the current word exist in the corpus, check if its probability is higher than the language model prediction (if exist)
            corrected_prediction.append(predicted_words[i])
            
        # update word1 and word2 for next iteration    
        word1=word2
        word2=new_word2
    #print("Debug : word1=", word1, " word2=", word2," corrected_prediction=" , corrected_prediction)    
            
# Uncomment to remove debug/unit test    
#print("True transcription:         ", label) 
#print("Model raw prediction:       ", prediction)
#print("Spell corrected prediction: ", ' '.join(corrected_prediction))


# Comment to remove debug/unit test 
print("\n\nUnit Test  transcription:         ", label) 
print("Unit Test Model raw prediction:       ", " ".join(predicted_words))
print("Unit Test Spell corrected prediction: ", ' '.join(corrected_prediction))