import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np
import librosa
import argparse
from datetime import datetime
import json
import os
import sys
import time
import scipy
import numpy.random as random
# import pyrubberband
import math

from speech_embedding import mix_generator
from speech_embedding.emb_data_generator import trim_silence, stft

# tf.keras.utils.Sequence
class DataGenerator:
    'Generates data for Keras'
    def __init__(self, speech_filenames, reverb_filenames, noise_filenames=None,
                      speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                      sample_rate=16000, 
                      batch_size=32,
                      seq_len=500, num_classes=250,
                      shuffle=True, 
                      in_memory=0.0, augment_speech=False,inject_noise=False, augment_reverb=False):
        
        self.speech_filenames = speech_filenames
        self.reverb_filenames = reverb_filenames
        self.noise_filenames = noise_filenames
        
        if speech_data_holder:
            self.speech_data_holder = speech_data_holder
        else:
            self.speech_data_holder = {}
            
        if reverb_data_holder:
            self.reverb_data_holder = reverb_data_holder
        else:
            self.reverb_data_holder = {}
        
        if noise_data_holder:
            self.noise_data_holder = noise_data_holder
        else:
            self.noise_data_holder = {}
        
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.in_memory = in_memory
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment_speech = augment_speech
        self.augment_reverb=augment_reverb
        self.inject_noise = inject_noise
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.speech_filenames) * len(self.reverb_filenames) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        idx0, idx1 = np.meshgrid(np.arange(len(self.speech_filenames)), 
                                 np.arange(len(self.reverb_filenames)))
        idx0 = idx0.reshape((-1))
        idx1 = idx1.reshape((-1))
        print(idx0.shape, idx1.shape)
        self.indexes = list(zip(idx0, idx1))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # in-place shuffle

    def get_speech_audio(self, speech_filename=None, idx=0):
        if speech_filename is None:
            speech_filename = self.speech_filenames[int(idx)]
        if speech_filename in self.speech_data_holder:
            speech_audio = self.speech_data_holder[speech_filename]
        else:
            speech_audio, _ = librosa.core.load(speech_filename, sr=self.sample_rate, mono=True)
            if len(self.speech_data_holder.keys()) < self.in_memory * len(self.speech_filenames):
                self.speech_data_holder[speech_filename] = speech_audio  
        speaker = speech_filename.split("/")[-1].split("_")[0]
        if speaker[0]=="f":
            number = int(speaker[1:])
        else:
            number = int(speaker[1:]) + 10
        return speech_audio, number, speech_filename

    def get_reverb_audio(self, reverb_filename=None, idx=0):
        if reverb_filename is None:
            reverb_filename = self.reverb_filenames[int(idx)]
        if reverb_filename in self.reverb_data_holder:
            reverb_audio = self.reverb_data_holder[reverb_filename]
        else:
            reverb_audio, _ = librosa.core.load(reverb_filename, sr=self.sample_rate, mono=True)
            if len(self.reverb_data_holder.keys()) < self.in_memory * len(self.reverb_filenames):
                self.reverb_data_holder[reverb_filename] = reverb_audio
        number = int(reverb_filename.split("/")[-1].split("_")[0][1:])
        return reverb_audio, number, reverb_filename
    
    def get_noise_audio(self, noise_filename=None, idx=0):
        if noise_filename is None:
            noise_filename = self.noise_filenames[int(idx)]
        if noise_filename in self.noise_data_holder:
            noise = self.noise_data_holder[noise_filename]
        else:
            noise, _ = librosa.core.load(noise_filename, sr=self.sample_rate, mono=True)
            if len(self.noise_data_holder.keys()) < self.in_memory * len(self.noise_filenames):
                self.noise_data_holder[noise_filename] = noise
        return noise, noise_filename
    
    def __iter__(self):
        input_list = []
        target_list = []
        
        while True:
            for idx in self.indexes:
                speech_filename_1 = self.speech_filenames[int(idx[0])]
                reverb_filename = self.reverb_filenames[int(idx[1])]
                if self.inject_noise:
                    idx = np.random.randint(len(self.noise_filenames))
                    noise_filename = self.noise_filenames[int(idx)]

                speech_audio_1, speaker_id_1, _ = self.get_speech_audio(speech_filename=speech_filename_1)
                speech_audio_1, _ = trim_silence(speech_audio_1)
                
                if self.seq_len > 0:
                    cut = int(np.random.randint(0, len(speech_audio_1)-int(self.seq_len), 1))
                    speech_audio_1 = speech_audio_1[cut:cut+int(self.seq_len)]
                
                reverb_audio, number, _ = self.get_reverb_audio(reverb_filename=reverb_filename)
                
                if self.inject_noise:
                    noise, _ = self.get_noise_audio(noise_filename=noise_filename)
                else:
                    noise = None

                noisy_audio_1, new_speech_audio, new_reverb, pre_noisy_audio_1 = mix_generator.mix_reverb_noise(speech_audio_1, 
                                                                                     reverb_audio, 
                                                                               self.sample_rate, 
                                                                               noise=noise, 
                                                                               augment_speech=self.augment_speech,
                                                                               augment_reverb=self.augment_reverb)

                input_list.append(np.float32(noisy_audio_1[np.newaxis, :self.seq_len, np.newaxis]))
                target_list.append(np.float32(new_speech_audio[np.newaxis, :self.seq_len, np.newaxis]))

                if len(input_list)==self.batch_size:
                    input_arr = np.asarray(input_list)
                    target_arr = np.asarray(target_list)

                    yield input_arr, target_arr
                    
                    input_list=[]
                    target_list=[]
                    
            self.on_epoch_end()