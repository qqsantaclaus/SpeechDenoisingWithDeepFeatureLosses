import copy
import fnmatch
import os
import random
import re
import threading
import math

import librosa
import numpy as np
import tensorflow as tf
import json

import pickle
from numpy.random import permutation
from numpy.random import randint
import numpy as np
import pandas as pd
from scipy import signal
import scipy

from read_Audio_RIRs import *
from speech_embedding import mix_generator as emb_mix_generator

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


CLEAN_DATA_RANGE = {"gender": ['m', 'f'], 
                                "num": list(np.arange(1, 10)), 
                                "script": [1,2,3,4], 
                                "device": ['clean'], 
                                "scene": []}

CLEAN_TEST_DATA_RANGE = {"gender": ['m', 'f'], 
                                         "num": [10],
                                         "script": [5], 
                                         "device": ['clean'], 
                                         "scene": []}

CLEAN_FEMALE_DATA_RANGE = {"gender": ['f'], 
                                "num": list(np.arange(1, 10)), 
                                "script": [1,2,3,4], 
                                "device": ['clean'], 
                                "scene": []}

CLEAN_FEMALE_TEST_DATA_RANGE = {"gender": ['f'], 
                                         "num": [10],
                                         "script": [5], 
                                         "device": ['clean'], 
                                         "scene": []}

DIRECTORY="daps"

GENDERS = ["f", "m"]

NUMS = range(1, 11)

SCRIPTS = range(1, 6)

DEVICES = ["ipad", "ipadflat", "iphone"]

SCENES = ["office1", "office2", "confroom1", "confroom2", "livingroom1", "bedroom1", "balcony1"]

def query_joint_yield_pair(gender=None, num=None, script=None, device=None, 
            scene=None, directory=DIRECTORY, directory_produced=DIRECTORY, exam_ignored=True, randomized=False,
                     sample_rate=None):
    '''
    inputs are all lists
    '''
    if exam_ignored:
        filtered_gender = gender if gender else GENDERS
        filtered_num = num if num else NUMS
        filtered_script = script if script else SCRIPTS
        filtered_device = device if device else DEVICES
        filtered_scene = scene if scene else SCENES
    else:
        filtered_gender = [g for g in gender if g in GENDERS] if gender else GENDERS
        filtered_num = [n for n in num if n in NUMS] if num else NUMS
        filtered_script = [s for s in script if s in SCRIPTS] if script else SCRIPTS
        filtered_device = [d for d in device if d in DEVICES] if device else DEVICES
        filtered_scene = [s for s in scene if s in SCENES ] if scene else SCENES
    book = [ (g, n, st, d, s) for g in filtered_gender for n in filtered_num 
                for st in filtered_script for d in filtered_device for s in filtered_scene]
    if randomized:
        book = permutation(book)
    
    for (g, n, st, d, s) in book:
        filename = directory+"/"+d+"_"+s+"/"+g+str(n)+"_script"+str(st)+"_"+d+"_"+s+".wav" 
        produced_filename = directory_produced+"/clean/"+g+str(n)+"_script"+str(st)+"_clean.wav"
        try:
            # print(filename, produced_filename)
            input_audio, _ = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
            target_audio, _ = librosa.load(produced_filename, sr=sample_rate, mono=True, dtype=np.float32)
            input_audio = butter_highpass_filter(input_audio, 30, sample_rate, order=5)
            target_audio = butter_highpass_filter(target_audio, 30, sample_rate, order=5)
            input_audio = input_audio.reshape(-1, 1)
            target_audio = target_audio.reshape(-1, 1)
            yield input_audio, target_audio, {"gender": g, "num": n, "script": st, "device": d, "scene": s}
        except Exception as e: 
            print(e)
            continue

def query_joint_yield(gender=None, num=None, script=None, device=None, 
            scene=None, directory=DIRECTORY, exam_ignored=True, randomized=False):
    '''
    inputs are all lists
    '''
    if exam_ignored:
        filtered_gender = gender if gender else GENDERS
        filtered_num = num if num else NUMS
        filtered_script = script if script else SCRIPTS
        filtered_device = device if device else DEVICES
        filtered_scene = scene if scene else SCENES
    else:
        filtered_gender = [g for g in gender if g in GENDERS] if gender else GENDERS
        filtered_num = [n for n in num if n in NUMS] if num else NUMS
        filtered_script = [s for s in script if s in SCRIPTS] if script else SCRIPTS
        filtered_device = [d for d in device if d in DEVICES] if device else DEVICES
        filtered_scene = [s for s in scene if s in SCENES ] if scene else SCENES
    book = [ (g, n, st) for g in filtered_gender for n in filtered_num 
                for st in filtered_script]
    filenames = []
    
    estimated_totals = len(book)
    
    for (g, n, st) in book:
        filename = directory+"/clean/"+g+str(n)+"_script"+str(st)+"_clean"+".wav" 
        if os.path.exists(filename):
            filenames.append(filename)
            
    return filenames

def trim_silence(audio, threshold=0.05, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rms(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return ((audio[indices[0]:indices[-1]], indices)
            if indices.size else (audio[0:0], None))


# epsilon = 1e-9

# SR = 16000

# '''
# The STFT parameters were: 10 ms
# hop-size, 25 ms frame-length, and Hann window.
# '''

# N_FFT = int(SR * 0.025)
# SHIFT = int(SR * 0.010)

# '''
# # Frames by # channels
# '''
# def stft(y):
#     return np.abs(librosa.core.stft(np.squeeze(y), n_fft=N_FFT, hop_length=SHIFT)).T


def approx_binary_label(D):
    raw_score = (-1*D/20.0)
    exp_score = np.exp(raw_score)
    binary_label = exp_score / np.sum(exp_score)
    return binary_label

def extract_speaker_id(speech_filename):
    label = speech_filename.split("/")[-1][0:3]
    try:
        speaker_id = int(label[1:3])
        if label[0]=="f":
            speaker_id = speaker_id + 10
    except:
        speaker_id = -1
        
    if speaker_id>=0:
        speaker_binary_label = tf.keras.utils.to_categorical(speaker_id-1, num_classes=20)
    else:
        speaker_binary_label = np.zeros((20,))
    
    return speaker_id, speaker_binary_label

class MixGeneratorSpec_single:
    'Generates data for Keras'
    def __init__(self, speech_filenames, reverb_filenames, noise_filenames, sample_rate,
                      speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                      num_classes=200,
                      shuffle=True, augment_speech=False, augment_reverb=False, norm_volume=False,
                      inject_noise=False,
                      raw_stft_similarity_score = None,
                      norm_reverb_peak=True,
                      SNRdB=None,
                      in_memory=1.0,
                      cut_length=160000):
        'Initialization'
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

        self.sample_rate = sample_rate
        
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment_speech = augment_speech
        self.augment_reverb = augment_reverb
        self.norm_volume = norm_volume
        self.inject_noise = inject_noise
        self.norm_reverb_peak = norm_reverb_peak
        self.raw_stft_similarity_score = raw_stft_similarity_score
        self.SNRdB = SNRdB
        self.in_memory = in_memory
        self.cut_length = int(cut_length)
        
        self.epoch_index = 0
        
        self.on_epoch_end()
#         print(len(self.indexes))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        idx0, idx1 = np.meshgrid(np.arange(len(self.speech_filenames)), 
                                 np.arange(len(self.reverb_filenames)))
        idx0 = idx0.reshape((-1))
        idx1 = idx1.reshape((-1))
        self.indexes = list(zip(idx0, idx1))
        print("Epoch: " + str(self.epoch_index) + " speech: " + str(len(self.speech_filenames)) + " reveb: " + str(len(self.reverb_filenames)) + " num samples: " + str(len(self.indexes)) + "\n")
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # in-place shuffle
        self.epoch_index = self.epoch_index + 1
            
    def num_elements(self):
        return len(self.indexes)
    
    def get_speech_audio(self, speech_filename=None, idx=0):
        if speech_filename is None:
            speech_filename = self.speech_filenames[int(idx)]
        if speech_filename in self.speech_data_holder:
            speech_audio = self.speech_data_holder[speech_filename]
        else:
            speech_audio, _ = librosa.core.load(speech_filename, sr=self.sample_rate, mono=True, dtype=np.float32)
            if len(self.speech_data_holder.keys()) < self.in_memory * len(self.speech_filenames):
                self.speech_data_holder[speech_filename] = speech_audio        
        return speech_audio, speech_filename

    def get_reverb_audio(self, reverb_filename=None, idx=0):
        if reverb_filename is None:
            reverb_filename = self.reverb_filenames[int(idx)]
        if reverb_filename["filepath"] in self.reverb_data_holder:
            reverb_audio = self.reverb_data_holder[reverb_filename["filepath"]]
        else:
            reverb_audio, _ = librosa.core.load(reverb_filename["filepath"], sr=self.sample_rate, mono=True, dtype=np.float32)
            if len(self.reverb_data_holder.keys()) < self.in_memory * len(self.reverb_filenames):
                self.reverb_data_holder[reverb_filename["filepath"]] = reverb_audio
#         number = int(reverb_filename.split("/")[-1][1:4])
        number = reverb_filename["id"]
        return reverb_audio, number, reverb_filename
    
    def get_noise_audio(self, noise_filename=None, idx=0):
        if noise_filename is None:
            noise_filename = self.noise_filenames[int(idx)]
        if noise_filename in self.noise_data_holder:
            noise = self.noise_data_holder[noise_filename]
        else:
            noise, _ = librosa.core.load(noise_filename, sr=self.sample_rate, mono=True, dtype=np.float32)
            if len(self.noise_data_holder.keys()) < self.in_memory * len(self.noise_filenames):
                self.noise_data_holder[noise_filename] = noise
        return noise, noise_filename

    def __iter__(self):
        while True:
            for idx in self.indexes:
                speech_filename = self.speech_filenames[int(idx[0])]
                reverb_filename = self.reverb_filenames[int(idx[1])]
                
                speech_audio, _ = self.get_speech_audio(speech_filename)
                speech_audio, keep_indices = trim_silence(speech_audio)
               
                reverb_audio, number, _ = self.get_reverb_audio(reverb_filename)
                
                if self.shuffle:
                    rand_start = int(np.random.randint(0, len(speech_audio) - self.cut_length, 1))
                    speech_audio = speech_audio[rand_start:rand_start+self.cut_length]
                
                # inject noise here: /mnt/ilcompfbd1/user/jsu/reverb_tools_for_Generate_SimData/NOISE
                if self.inject_noise:
                    if np.random.uniform()<0.9:
                        noise_idx = int(np.random.randint(0, len(self.noise_filenames), 1))
                        noise, _ = self.get_noise_audio(idx=noise_idx)
                    else:
                        noise = emb_mix_generator.generate_gaussian_noise(len(speech_audio))
                else:
                    noise = None
                
                noisy_audio, speech_audio, _, pre_noisy_audio = emb_mix_generator.mix_reverb_noise(speech_audio, 
                                                                                   reverb_audio, 
                                                                                   sample_rate=self.sample_rate, 
                                                                                   noise=noise, 
                                                                                   augment_speech=self.augment_speech,
                                                                                   augment_reverb=self.augment_reverb,
                                                                                   norm_volume=self.norm_volume,
                                                                                   norm_reverb_peak=self.norm_reverb_peak,
                                                                                   SNRdB=self.SNRdB)
                
                if number<self.num_classes:
                    binary_label = tf.keras.utils.to_categorical(number, num_classes=self.num_classes)
                else:
                    if self.raw_stft_similarity_score is not None:
                        binary_label = self.raw_stft_similarity_score[number, :]                        
                    else:
                        binary_label = np.zeros((self.num_classes,))
                        
                speaker_id, speaker_binary_label = extract_speaker_id(speech_filename)
                
                string_label = reverb_filename["filepath"].split("/")[-1].split(".")[0]+"|"+speech_filename.split("/")[-1].split(".")[0]

                yield noisy_audio[:, np.newaxis], pre_noisy_audio[:, np.newaxis], speech_audio[:, np.newaxis], binary_label, speaker_binary_label, string_label
            
            self.on_epoch_end()

class MixGeneratorSpec_pair(MixGeneratorSpec_single):
    'Generates data for Keras'
    def __init__(self, speech_filenames, reverb_filenames, noise_filenames, sample_rate,
                      speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                      num_classes=200,
                      shuffle=True, augment_speech=False, augment_reverb=False, norm_volume=False,
                      inject_noise=False, raw_stft_similarity_score=None,
                      norm_reverb_peak=True,
                      SNRdB=None,
                      in_memory=1.0,
                      cut_length=160000):
        'Initialization'
        super().__init__(speech_filenames, reverb_filenames, noise_filenames, sample_rate,
                          speech_data_holder=speech_data_holder, 
                          reverb_data_holder=reverb_data_holder, 
                          noise_data_holder=noise_data_holder,
                          num_classes=num_classes,
                          shuffle=shuffle, augment_speech=augment_speech, augment_reverb=augment_reverb, norm_volume=norm_volume,
                          inject_noise=inject_noise, raw_stft_similarity_score=raw_stft_similarity_score, 
                          norm_reverb_peak=norm_reverb_peak,
                          SNRdB=SNRdB,
                          in_memory=in_memory,
                          cut_length=cut_length)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        idx1, idx2 = np.meshgrid(np.arange(len(self.reverb_filenames)),
                                 np.arange(len(self.reverb_filenames)))
        idx1 = idx1.reshape((-1))
        idx2 = idx2.reshape((-1))
        print(len(self.speech_filenames), len(self.reverb_filenames), idx1.shape, idx2.shape)
        self.indexes = list(zip(idx1, idx2))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # in-place shuffle
        self.epoch_index = self.epoch_index + 1
        
    def __iter__(self):
        while True:
            for idx in self.indexes:
                speech_audio, speech_filename = self.get_speech_audio(idx=np.random.randint(0, len(self.speech_filenames), 1))
                speech_audio, keep_indices = trim_silence(speech_audio)
                
                if self.shuffle:
                    rand_start = int(np.random.randint(0, len(speech_audio) - self.cut_length, 1))
                    speech_audio = speech_audio[rand_start:rand_start+self.cut_length]
                
                reverb_audio, number, reverb_filename = self.get_reverb_audio(idx=idx[0])
                reverb_audio_2, number_2, reverb_filename_2 = self.get_reverb_audio(idx=idx[1])
                
                # inject noise here: /mnt/ilcompfbd1/user/jsu/reverb_tools_for_Generate_SimData/NOISE
                if self.inject_noise:
                    if np.random.uniform()<0.95:
                        noise, _ = self.get_noise_audio(idx=np.random.randint(0, len(self.noise_filenames), 1))
                    else:
                        noise = emb_mix_generator.generate_gaussian_noise(len(speech_audio))
                    if np.random.uniform()<0.95:
                        noise_2, _ = self.get_noise_audio(idx=np.random.randint(0, len(self.noise_filenames), 1))
                    else:
                        noise_2 = emb_mix_generator.generate_gaussian_noise(len(speech_audio))
                else:
                    noise = None
                    noise_2 = None
                
                noisy_audio, speech_audio, _, _ = emb_mix_generator.mix_reverb_noise(speech_audio, 
                                                                                   reverb_audio, 
                                                                                   sample_rate=self.sample_rate, 
                                                                                   noise=noise, 
                                                                                   augment_speech=self.augment_speech,
                                                                                   augment_reverb=self.augment_reverb,
                                                                                   norm_volume=self.norm_volume,
                                                                                   norm_reverb_peak=self.norm_reverb_peak,
                                                                                   SNRdB=self.SNRdB)
                
                noisy_audio_2, _, _, pre_noisy_audio_2 = emb_mix_generator.mix_reverb_noise(speech_audio, 
                                                                      reverb_audio_2, 
                                                                      sample_rate=self.sample_rate, 
                                                                      noise=noise_2, 
                                                                      augment_speech=False,
                                                                      augment_reverb=self.augment_reverb,
                                                                      norm_volume=self.norm_volume,
                                                                      norm_reverb_peak=self.norm_reverb_peak,
                                                                      SNRdB=self.SNRdB)
                
                if number<self.num_classes:
                    binary_label = tf.keras.utils.to_categorical(number, num_classes=self.num_classes)
                else:
                    if self.raw_stft_similarity_score is not None:
                        binary_label = self.raw_stft_similarity_score[number, :]                        
                    else:
                        binary_label = np.zeros((self.num_classes,))
                
                if number_2<self.num_classes:
                    binary_label_2 = tf.keras.utils.to_categorical(number_2, num_classes=self.num_classes)
                else:
                    if self.raw_stft_similarity_score is not None:
                        binary_label_2 = self.raw_stft_similarity_score[number_2, :]                        
                    else:
                        binary_label_2 = np.zeros((self.num_classes,))
                        
                string_label = reverb_filename["filepath"].split("/")[-1].split(".")[0]+"|"+speech_filename.split("/")[-1].split(".")[0]
                string_label_2 = reverb_filename_2["filepath"].split("/")[-1].split(".")[0]+"|"+speech_filename.split("/")[-1].split(".")[0]
                
                yield noisy_audio[:, np.newaxis], noisy_audio_2[:, np.newaxis], pre_noisy_audio_2[:, np.newaxis], speech_audio[:, np.newaxis], binary_label, binary_label_2, string_label, string_label_2
            
            self.on_epoch_end()

class DataReader(object):
    '''Generic background audio reader that preprocesses audio files
    and en
    s them into a TensorFlow queue.'''
    def __init__(self,
                 directory,
                 coord,
                 sample_size,
                 hint_size,
                 target_size,
                 sample_rate,
                 random_crop=True,
                 queue_size=32,
                 data_range=CLEAN_DATA_RANGE,
                 test_data_range=CLEAN_TEST_DATA_RANGE,
                 disc_thread_enabled=True,
                 spec_generator=None,
                 use_label_class=False,
                 hint_window=256,
                 inject_noise=False,
                 augment_reverb=False,
                 augment_speech=True,
                 norm_volume=False,
                 stft_similarity=None,
                 norm_reverb_peak=True):
        self.directory = os.path.abspath(directory)
        print(self.directory)
        
        self.data_range = data_range
        self.test_data_range = test_data_range
        self.coord = coord
        self.sample_size = sample_size
        self.random_crop = random_crop
        self.target_size = target_size
        self.hint_size = hint_size
        self.sample_rate = sample_rate
        self.silence_threshold = 0.15
        self.disc_thread_enabled = disc_thread_enabled
        self.use_label_class = use_label_class
        self.hint_window = hint_window
        self.inject_noise = inject_noise
        self.augment_reverb = augment_reverb
        self.augment_speech = augment_speech
        self.norm_volume = norm_volume
        self.norm_reverb_peak = norm_reverb_peak
        
        if use_label_class and augment_reverb:
            raise ValueError("Reverbs can not be augmented when class label is used for conditioning.")
        
        self.stft_similarity = stft_similarity
        if self.stft_similarity is not None:
            raw_score = (-1*self.stft_similarity[:, :200]/20.0)
            exp_score = np.exp(raw_score)
            self.raw_stft_similarity_score = exp_score / np.sum(exp_score, axis=-1, keepdims=True)
        else:
            self.raw_stft_similarity_score = None
        
        self.spec_generator = spec_generator
        
        self.train_filenames = query_joint_yield(gender=data_range["gender"], 
                                                 num=data_range["num"], 
                                                 script=data_range["script"],
                                                 device=data_range["device"], 
                                                 scene=data_range["scene"], 
                                                 directory=directory, 
                                                 exam_ignored=True, 
                                                 randomized=True)

        self.test_filenames = query_joint_yield(gender=test_data_range["gender"], 
                                                 num=test_data_range["num"], 
                                                 script=test_data_range["script"],
                                                 device=test_data_range["device"], 
                                                 scene=test_data_range["scene"], 
                                                 directory=directory, 
                                                 exam_ignored=True, 
                                                 randomized=True)

        #%% Reverb audio files
        # 0-199, 271-595 Train
        # 200-270 Test
        # "h233"/232 is missing
        self.class_dict, self.classes = get_Audio_RIR_classes("/home/code-base/runtime/experiments/weakly-aligned-denoising/rir_classes.json")
        
        self.reverb_train_filenames, self.reverb_train_data_holder=read_Audio_RIRs(sr=self.sample_rate, 
                                                                         class_dict=self.class_dict,
                                                                         subset_range=list(range(200))+list(range(271, 596)))
                                                                    
        self.reverb_test_filenames, self.reverb_test_data_holder=read_Audio_RIRs(sr=self.sample_rate, 
                                                                         class_dict=self.class_dict,
                                                                         subset_range=list(range(200, 271)))
        
        self.noise_filenames, self.noise_data_holder = read_noise(sr=self.sample_rate, 
                                                                  root="/trainman-mount/trainman-storage-420a420f-b7a2-4445-abca-0081fc7108ca/subnoises", preload=False)

        ######
        
        self.threads = []
        
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.speaker_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 20))
        self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32', 'float32', 'float32', 'float32'],
                                         shapes=[(None, 1), (None, hint_size), (None, 20),  (None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder, 
                                           self.hint_placeholder,
                                           self.speaker_hint_placeholder,
                                           self.target_placeholder])
        
        self.disc_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.disc_ref_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.disc_queue = tf.PaddingFIFOQueue(16,
                                         ['float32', 'float32'],
                                         shapes=[(None, 1), (None, 1)])
        self.disc_enqueue = self.disc_queue.enqueue([self.disc_placeholder, 
                                                     self.disc_ref_placeholder])
        
        """For val set"""
        self.test_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_speaker_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 20))
        self.test_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_queue = tf.PaddingFIFOQueue(40,
                                               ['float32', 'float32', 'float32', 'float32'],
                                               shapes=[(None, 1), (None, hint_size), (None, 20), (None, 1)])
        self.test_enqueue = self.test_queue.enqueue([self.test_placeholder, 
                                                     self.test_hint_placeholder,
                                                     self.test_speaker_hint_placeholder,
                                                     self.test_target_placeholder])
        """For test set"""
        self.test_ext_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_ext_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_ext_speaker_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 20))
        self.test_ext_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_ext_queue = tf.PaddingFIFOQueue(40,
                                               ['float32', 'float32', 'float32', 'float32'],
                                               shapes=[(None, 1), (None, hint_size), (None, 20), (None, 1)])
        self.test_ext_enqueue = self.test_ext_queue.enqueue([self.test_ext_placeholder, 
                                                     self.test_ext_hint_placeholder,
                                                     self.test_ext_speaker_hint_placeholder,
                                                     self.test_ext_target_placeholder])
        """For real set"""
        self.test_real_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_real_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_real_speaker_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 20))
        self.test_real_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_real_queue = tf.PaddingFIFOQueue(16,
                                               ['float32', 'float32', 'float32', 'float32'],
                                               shapes=[(None, 1), (None, hint_size), (None, 20), (None, 1)])
        self.test_real_enqueue = self.test_real_queue.enqueue([self.test_real_placeholder, 
                                                     self.test_real_hint_placeholder,
                                                     self.test_real_speaker_hint_placeholder,
                                                     self.test_real_target_placeholder])
        """For inference"""
        self.infer_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.infer_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.infer_speaker_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 20))
        self.infer_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.infer_class_placeholder = tf.placeholder(dtype=tf.string, shape=None)
        self.infer_queue = tf.PaddingFIFOQueue(queue_size,
                                               ['float32', 'float32', 'float32', 'float32', 'string'],
                                               shapes=[(None, 1), (None, hint_size), (None, 20),  (None, 1), (None,)])
        self.infer_enqueue = self.infer_queue.enqueue([self.infer_placeholder, 
                                                       self.infer_hint_placeholder,
                                                       self.infer_speaker_hint_placeholder,
                                                       self.infer_target_placeholder,
                                                       self.infer_class_placeholder])
        

    def dequeue(self, num_elements):
        return self.queue.dequeue_many(num_elements)

    def dequeue_test(self, num_elements):
        return self.test_queue.dequeue_many(num_elements)

    def dequeue_test_ext(self, num_elements):
        return self.test_ext_queue.dequeue_many(num_elements)
    
    def dequeue_test_real(self, num_elements):
        return self.test_real_queue.dequeue_many(num_elements)
    
    def dequeue_disc(self, num_elements):
        return self.disc_queue.dequeue_many(num_elements)
    
    def dequeue_infer(self, num_elements):
        return self.infer_queue.dequeue_many(num_elements)
    
    # During training, select spectrogram frames centered around waveform piece if random_crop enbaled; otherwise randomly
    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        mix_generator = MixGeneratorSpec_single(self.train_filenames, self.reverb_train_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=True,
                                          augment_speech=self.augment_speech,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=self.augment_reverb,
                                          norm_volume=self.norm_volume,
                                          norm_reverb_peak=self.norm_reverb_peak,
                                          cut_length=self.sample_size*5.0)

        print("Loading Data...")
        for input_waveform, _, target_waveform, binary_label, speaker_binary_label, _ in mix_generator:
            if self.coord.should_stop():
                stop = True
                break

            # padding
            lag = self.target_size
#             random_start = int(randint(0, lag, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
                total_frames = input_spec.shape[0]
#                 print(total_frames)
            elif self.use_label_class and self.hint_size==len(binary_label):
                hint_piece = np.reshape(binary_label, (1, self.hint_size))
            else:
                hint_piece = np.zeros((1, self.hint_size))
                
            speaker_hint_piece = np.reshape(speaker_binary_label, (1, -1))
            
#                 print(np.argmax(binary_label), input_waveform.shape, target_waveform.shape, input_spec.shape if self.spec_generator else 0)

            # input_waveform and target_waveform are now of same length, and with 0-padding in front
            if not self.random_crop:
                while len(input_waveform) > self.sample_size:
                    piece = input_waveform[:self.sample_size, :]
                    input_waveform = input_waveform[lag:, :]

                    start = int((self.sample_size-self.target_size)/2)
                    target_piece = target_waveform[start:start+self.target_size, :]
                    target_waveform = target_waveform[lag:, :]

                    if self.spec_generator:
                        if self.hint_window<=0:
                            hint_piece = input_spec
                        else:
                            random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
                            hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: piece, 
                                        self.hint_placeholder: hint_piece,
                                        self.speaker_hint_placeholder: speaker_hint_piece,
                                        self.target_placeholder: target_piece})
            else:
                length = input_waveform.shape[0]
                num_pieces = 1
#                 print(num_pieces)
                indices = randint(0, length-self.sample_size, num_pieces) 
                
#                 if self.spec_generator:
#                     spec_indices = librosa.core.samples_to_frames(indices + int(self.sample_size/2), hop_length=self.spec_generator.shift, n_fft=self.spec_generator.n_fft)
                
                for i in range(num_pieces):
                    idx = indices[i]
                    central = int(idx + self.sample_size/2-self.target_size/2)
                    piece = input_waveform[idx:idx+self.sample_size, :]
                    target_piece = target_waveform[central:central+self.target_size, :]

                    if self.spec_generator:
                        if self.hint_window<=0:
                            hint_piece = input_spec
                        else:
#                             random_spec = spec_indices[i]
#                             random_shift = randint(-int(self.hint_window/4), int(self.hint_window/4), 1)
#                             random_start = max(0, int(random_spec - self.hint_window/2 + random_shift))
#                             random_end = min(int(random_spec + self.hint_window/2 + random_shift), total_frames)
#                             hint_piece = input_spec[random_start:random_end, :]
                            random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
                            hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

                    sess.run(self.enqueue,
                         feed_dict={self.sample_placeholder: piece, 
                                    self.hint_placeholder: hint_piece,
                                    self.speaker_hint_placeholder: speaker_hint_piece,
                                    self.target_placeholder: target_piece})

    # During testing, use entire audio file for spectrogram frames
    def thread_test(self, sess):
        stop = False
        infer_sample_size = self.target_size * 3 + self.sample_size
        # Go through the dataset multiple times
        mix_generator = MixGeneratorSpec_single(self.test_filenames, self.reverb_train_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=True,
                                          augment_speech=False,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=False,
                                          norm_volume=self.norm_volume,
                                          raw_stft_similarity_score=self.raw_stft_similarity_score,
                                          norm_reverb_peak=False,
                                          SNRdB=[20],
                                          cut_length=infer_sample_size)

        for input_waveform, _, target_waveform, binary_label, speaker_binary_label, number in mix_generator:
            if self.coord.should_stop():
                stop = True
                break

            print("test:", np.argmax(binary_label), number, input_waveform.shape, target_waveform.shape)

#                 target_waveform, keep_indices = trim_silence(target_waveform[:, 0],
#                                                   self.silence_threshold)
#                 target_waveform = target_waveform.reshape(-1, 1)
#                 if target_waveform.size == 0:
#                     print("Warning: {} was ignored as it contains only "
#                           "silence. Consider decreasing trim_silence "
#                           "threshold, or adjust volume of the audio."
#                           .format(id_dict))

#                 input_waveform = input_waveform[keep_indices[0]:keep_indices[-1], :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
                print("test: from spec generator", input_spec.shape)
            elif self.use_label_class and self.hint_size==len(binary_label):
                hint_piece = np.reshape(binary_label, (1, self.hint_size))
                print("test: from binary label", hint_piece.shape)
            else:
                hint_piece = np.zeros((1, self.hint_size))
                print("test: from dummy zeros", hint_piece.shape)
                
            speaker_hint_piece = np.reshape(speaker_binary_label, (1, -1))

#             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]

            piece = input_waveform[:infer_sample_size, :]
            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]

            if self.spec_generator:
#                     if self.hint_window<=0:
                hint_piece = input_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

            sess.run(self.test_enqueue,
                     feed_dict={self.test_placeholder: piece, 
                                self.test_hint_placeholder: hint_piece,
                                self.test_speaker_hint_placeholder: speaker_hint_piece,
                                self.test_target_placeholder: target_piece})
    
    # During testing, use entire audio file for spectrogram frames
    def thread_test_ext(self, sess):
        stop = False
        infer_sample_size = self.target_size * 3 + self.sample_size
        # Go through the dataset multiple times
        mix_generator = MixGeneratorSpec_single(self.test_filenames, self.reverb_test_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=True,
                                          augment_speech=False,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=False,
                                          norm_volume=self.norm_volume,
                                          raw_stft_similarity_score=self.raw_stft_similarity_score,
                                          norm_reverb_peak=False,
                                          SNRdB=[20],
                                          cut_length=infer_sample_size)

        for input_waveform, _, target_waveform, binary_label, speaker_binary_label, number in mix_generator:
            if self.coord.should_stop():
                stop = True
                break

            print("Ext test:", np.argmax(binary_label), number, input_waveform.shape, target_waveform.shape)

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
            else:
                hint_piece = np.zeros((1, self.hint_size))
                
            speaker_hint_piece = np.reshape(speaker_binary_label, (1, -1))

#             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]
            piece = input_waveform[:infer_sample_size, :]
            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]

            if self.spec_generator:
#                     if self.hint_window<=0:
                hint_piece = input_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

            sess.run(self.test_ext_enqueue,
                     feed_dict={self.test_ext_placeholder: piece, 
                                self.test_ext_hint_placeholder: hint_piece,
                                self.test_ext_speaker_hint_placeholder: speaker_hint_piece,
                                self.test_ext_target_placeholder: target_piece})

    # During testing, use entire audio file for spectrogram frames
    def thread_test_real(self, sess):
        stop = False
        # Go through the dataset multiple times
        mix_generator = query_joint_yield_pair(gender=self.test_data_range["gender"], num=self.test_data_range["num"], 
                                               script=self.test_data_range["script"], device=None, 
                                               scene=None, directory=self.directory, directory_produced=self.directory, 
                                               exam_ignored=False, 
                                               randomized=True,
                                               sample_rate=self.sample_rate)

        for input_waveform, target_waveform, book in mix_generator:
            if self.coord.should_stop():
                stop = True
                break

            print("Real test:", book, input_waveform.shape, target_waveform.shape)

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
            else:
                hint_piece = np.zeros((1, self.hint_size))
            
            speaker_id, speaker_binary_label = extract_speaker_id(book["gender"]+str(book["num"]))
            speaker_hint_piece = np.reshape(speaker_binary_label, (1, -1))

            count = 0
            max_count = 10

            random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
            input_waveform = input_waveform[random_start:, :]
            target_waveform = target_waveform[random_start:, :]

            while len(input_waveform) > self.sample_size and count<max_count:
                count = count + 1

                piece = input_waveform[:self.sample_size, :]
                input_waveform = input_waveform[int(self.target_size/2):, :]

                start = int((self.sample_size-self.target_size)/2)
                target_piece = target_waveform[start:start+self.target_size, :]
                target_waveform = target_waveform[int(self.target_size/2):, :]

                if self.spec_generator:
#                     if self.hint_window<=0:
                    hint_piece = input_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

                sess.run(self.test_real_enqueue,
                         feed_dict={self.test_real_placeholder: piece, 
                                    self.test_real_hint_placeholder: hint_piece,
                                    self.test_real_speaker_hint_placeholder: speaker_hint_piece,
                                    self.test_real_target_placeholder: target_piece})


    # During inference, use entire audio file for spectrogram frames
    def thread_infer(self, sess, start_idx=0, end_idx=271):
        stop = False
        infer_sample_size = self.target_size * 11 + self.sample_size
        # Go through the dataset multiple times
        full_reverb_filenames=self.reverb_train_filenames+self.reverb_test_filenames
        my_reverb_filenames=full_reverb_filenames[start_idx:end_idx]
        mix_generator = MixGeneratorSpec_single(self.test_filenames, my_reverb_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=False,
                                          augment_speech=False,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=False,
                                          norm_volume=self.norm_volume,
                                          raw_stft_similarity_score=self.raw_stft_similarity_score,
                                          norm_reverb_peak=False,
                                          SNRdB=[20],
                                          cut_length=infer_sample_size)

        for input_waveform, _, target_waveform, binary_label, speaker_binary_label, number in mix_generator:
            if mix_generator.epoch_index>1:
                print("All finished")
                stop = True
                self.infer_queue.close(cancel_pending_enqueues=False)
                break
                
#             input_waveform = input_waveform[170000:]
#             target_waveform = target_waveform[170000:]

            if self.coord.should_stop():
                stop = True
                break
#                 target_waveform, keep_indices = trim_silence(target_waveform[:, 0],
#                                                   self.silence_threshold)
#                 target_waveform = target_waveform.reshape(-1, 1)
#                 if target_waveform.size == 0:
#                     print("Warning: {} was ignored as it contains only "
#                           "silence. Consider decreasing trim_silence "
#                           "threshold, or adjust volume of the audio."
#                           .format(id_dict))

#                 input_waveform = input_waveform[keep_indices[0]:keep_indices[-1], :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
            elif self.use_label_class and self.hint_size==len(binary_label):
                hint_piece = np.reshape(binary_label, (1, self.hint_size))
            else:
                hint_piece = np.zeros((1, self.hint_size))
                
            speaker_hint_piece = np.reshape(speaker_binary_label, (1, -1))

#             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]

            piece = input_waveform[:infer_sample_size, :]

            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]

            if self.spec_generator:
#                     if self.hint_window<=0:
                hint_piece = input_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

            sess.run(self.infer_enqueue,
                     feed_dict={self.infer_placeholder: piece, 
                                self.infer_hint_placeholder: hint_piece,
                                self.infer_speaker_hint_placeholder: speaker_hint_piece,
                                self.infer_target_placeholder: target_piece,
                                self.infer_class_placeholder: np.array([number])})

    # During inference, use entire audio file for spectrogram frames
    def thread_infer_real(self, sess):
        stop = False
        infer_sample_size = self.target_size * 11 + self.sample_size
        # Go through the dataset multiple times
        mix_generator = query_joint_yield_pair(gender=self.test_data_range["gender"], num=self.test_data_range["num"], 
                                               script=self.test_data_range["script"], device=None, 
                                               scene=None, directory=self.directory, directory_produced=self.directory, 
                                               exam_ignored=False, 
                                               randomized=False,
                                               sample_rate=self.sample_rate)

        for input_waveform, target_waveform, book in mix_generator:
            if self.coord.should_stop():
                stop = True
                break
            
            target_waveform, keep_indices = trim_silence(target_waveform[:, 0])
            target_waveform = target_waveform.reshape(-1, 1)
            input_waveform = input_waveform[keep_indices[0]:keep_indices[-1], :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:infer_sample_size, :])
            else:
                hint_piece = np.zeros((1, self.hint_size))
                
            speaker_id, speaker_binary_label = extract_speaker_id(book["gender"]+str(book["num"]))
            speaker_hint_piece = np.reshape(speaker_binary_label, (1, -1))

            piece = input_waveform[:infer_sample_size, :]

            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]

            if self.spec_generator:
#                     if self.hint_window<=0:
                hint_piece = input_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]
           
            label = book["device"]+"_"+book["scene"]+"|"+book["gender"]+str(book["num"])+"_script"+str(book["script"])       
            sess.run(self.infer_enqueue,
                     feed_dict={self.infer_placeholder: piece, 
                                self.infer_hint_placeholder: hint_piece,
                                self.infer_speaker_hint_placeholder: speaker_hint_piece,
                                self.infer_target_placeholder: target_piece,
                                self.infer_class_placeholder: np.array([label])})

    def thread_disc(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            mix_generator = MixGeneratorSpec_single(self.test_filenames, self.reverb_train_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                              num_classes=200,
                                              shuffle=True,
                                              augment_speech=False,
                                              inject_noise=self.inject_noise)
            
            for input_waveform, _, target_waveform, binary_label, _ in mix_generator:
                if self.coord.should_stop():
                    stop = True
                    break
                
                while len(target_waveform) > self.sample_size:
                    target_piece = target_waveform[:self.target_size, :]
                    input_piece = input_waveform[:self.target_size, :]
                    
                    target_waveform = target_waveform[int(self.target_size/2):, :]
                    input_waveform = input_waveform[int(self.target_size/2):, :]
                    
                    sess.run(self.disc_enqueue,
                             feed_dict={self.disc_placeholder: target_piece, 
                                        self.disc_ref_placeholder: input_piece})

    def start_threads(self, sess, n_threads=1):
        for i in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
            print("Generator train data loader thread ("+str(i+1)+"/"+str(n_threads)+") starts.")
        thread = threading.Thread(target=self.thread_test, args=(sess,))
        thread.daemon = True  # Thread will close when parent quits.
        thread.start()
        self.threads.append(thread)
        print("Generator val data loader thread (1/1) starts.")
        thread = threading.Thread(target=self.thread_test_ext, args=(sess,))
        thread.daemon = True  # Thread will close when parent quits.
        thread.start()
        self.threads.append(thread)
        print("Generator test data loader thread (1/1) starts.")
#         thread = threading.Thread(target=self.thread_test_real, args=(sess,))
#         thread.daemon = True  # Thread will close when parent quits.
#         thread.start()
#         self.threads.append(thread)
#         print("Generator test real data loader thread (1/1) starts.")
        if self.disc_thread_enabled:
            thread = threading.Thread(target=self.thread_disc, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
            print("Discriminator data loader thread (1/1) starts.")
        return self.threads
    
    def start_infer_threads(self, sess, sim=True, n_threads=1):
        if sim:
            total_num = len(self.reverb_train_filenames+self.reverb_test_filenames)
            piece = int(math.ceil(total_num/n_threads))
            for i in range(n_threads):
                thread = threading.Thread(target=self.thread_infer, args=(sess, i*piece, min((i+1)*piece, total_num)))
                thread.daemon = True  # Thread will close when parent quits.
                thread.start()
                self.threads.append(thread)
                print("Generator sim infer data loader thread ("+str(i+1)+"/"+str(n_threads)+") starts.")
        else:
            thread = threading.Thread(target=self.thread_infer_real, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
            print("Generator real infer data loader thread (1/1) starts.")
        return self.threads

    def output_audio(self, path, wav):
        librosa.output.write_wav(path, wav, self.sample_rate)

class DataReader_pair(DataReader):
    '''Generic background audio reader that preprocesses audio files
    and en
    s them into a TensorFlow queue.'''

    def __init__(self,
                 directory,
                 coord,
                 sample_size,
                 hint_size,
                 target_size,
                 sample_rate,
                 random_crop=True,
                 queue_size=32,
                 data_range=CLEAN_DATA_RANGE,
                 test_data_range=CLEAN_TEST_DATA_RANGE,
                 disc_thread_enabled=True,
                 spec_generator=None,
                 use_label_class=False,
                 hint_window=256,
                 inject_noise=False,
                 augment_reverb=False,
                 augment_speech=True,
                 norm_volume=False,
                 stft_similarity=None,
                 norm_reverb_peak=True):
        
        super().__init__(directory,
                         coord,
                         sample_size,
                         hint_size,
                         target_size,
                         sample_rate,
                         random_crop=random_crop,
                         queue_size=queue_size,
                         data_range=data_range,
                         test_data_range=test_data_range,
                         disc_thread_enabled=disc_thread_enabled,
                         spec_generator=spec_generator,
                         use_label_class=use_label_class,
                         hint_window=hint_window,
                         inject_noise=inject_noise,
                         augment_reverb=augment_reverb,
                         augment_speech=augment_speech,
                         norm_volume=norm_volume,
                         stft_similarity=stft_similarity,
                         norm_reverb_peak=norm_reverb_peak)
        
        ######
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.src_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.tgt_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)

        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32', 'float32', 'float32', 'float32'],
                                         shapes=[(None, 1), (None, hint_size), (None, hint_size),  (None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder, 
                                           self.src_hint_placeholder,
                                           self.tgt_hint_placeholder,
                                           self.target_placeholder])

        """For val set"""
        self.test_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_src_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_tgt_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_queue = tf.PaddingFIFOQueue(8,
                                              ['float32', 'float32', 'float32', 'float32'],
                                              shapes=[(None, 1), (None, hint_size), (None, hint_size),  (None, 1)])
        self.test_enqueue = self.test_queue.enqueue([self.test_placeholder, 
                                                     self.test_src_hint_placeholder,
                                                     self.test_tgt_hint_placeholder,
                                                     self.test_target_placeholder])
        """For test set"""
        self.test_ext_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_ext_src_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_ext_tgt_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_ext_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_ext_queue = tf.PaddingFIFOQueue(8,
                                                  ['float32', 'float32', 'float32', 'float32'],
                                                  shapes=[(None, 1), (None, hint_size), (None, hint_size),  (None, 1)])
        self.test_ext_enqueue = self.test_ext_queue.enqueue([self.test_ext_placeholder, 
                                                     self.test_ext_src_hint_placeholder,
                                                     self.test_ext_tgt_hint_placeholder,
                                                     self.test_ext_target_placeholder])
        """For inference"""
        self.infer_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.infer_src_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.infer_tgt_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.infer_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.infer_src_class_placeholder = tf.placeholder(dtype=tf.string, shape=None)
        self.infer_tgt_class_placeholder = tf.placeholder(dtype=tf.string, shape=None)
        self.infer_queue = tf.PaddingFIFOQueue(queue_size,
                                               ['float32', 'float32', 'float32', 'float32', 'string', 'string'],
                                               shapes=[(None, 1), (None, hint_size), (None, hint_size),  (None, 1), (None,), (None,)])
        self.infer_enqueue = self.infer_queue.enqueue([self.infer_placeholder, 
                                                       self.infer_src_hint_placeholder,
                                                       self.infer_tgt_hint_placeholder,
                                                       self.infer_target_placeholder,
                                                       self.infer_src_class_placeholder,
                                                       self.infer_tgt_class_placeholder])
        """For real set"""
        self.test_real_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_real_src_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_real_tgt_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_real_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_real_queue = tf.PaddingFIFOQueue(8,
                                               ['float32', 'float32', 'float32', 'float32'],
                                               shapes=[(None, 1), (None, hint_size), (None, hint_size),  (None, 1)])
        self.test_real_enqueue = self.test_real_queue.enqueue([self.test_real_placeholder, 
                                                     self.test_real_src_hint_placeholder,
                                                     self.test_real_tgt_hint_placeholder,
                                                     self.test_real_target_placeholder])

    # During training, select spectrogram frames centered around waveform piece if random_crop enbaled; otherwise randomly
    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        mix_generator = MixGeneratorSpec_pair(self.train_filenames, self.reverb_train_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=True,
                                          augment_speech=self.augment_speech,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=self.augment_reverb,
                                          norm_volume=self.norm_volume,
                                          norm_reverb_peak=self.norm_reverb_peak,
                                          cut_length=self.sample_size*5.0)

        print("Loading Data...")
        for input_waveform, hint_waveform, target_waveform, _, src_binary_label, tgt_binary_label, _, _ in mix_generator:
            if self.coord.should_stop():
                stop = True
                break

            # padding
            lag = self.target_size
#             random_start = int(randint(0, lag, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
                src_total_frames = input_spec.shape[0]
                
                target_spec = self.spec_generator.__preprocess__(hint_waveform[:, 0])
                tgt_total_frames = target_spec.shape[0]
            elif self.use_label_class and self.hint_size==len(src_binary_label) and self.hint_size==len(tgt_binary_label):
                src_hint_piece = np.reshape(src_binary_label, (1, self.hint_size))
                tgt_hint_piece = np.reshape(tgt_binary_label, (1, self.hint_size))
            else:
                src_hint_piece = np.zeros((1, self.hint_size))
                tgt_hint_piece = np.zeros((1, self.hint_size))

#                 print(np.argmax(binary_label), input_waveform.shape, target_waveform.shape, input_spec.shape if self.spec_generator else 0)

            # input_waveform and target_waveform are now of same length, and with 0-padding in front
            if not self.random_crop:
                while len(input_waveform) > self.sample_size:
                    piece = input_waveform[:self.sample_size, :]
                    input_waveform = input_waveform[lag:, :]

                    start = int((self.sample_size-self.target_size)/2)
                    target_piece = target_waveform[start:start+self.target_size, :]
                    target_waveform = target_waveform[lag:, :]

                    if self.spec_generator:
                        if self.hint_window<=0:
                            src_hint_piece = input_spec
                            tgt_hint_piece = target_spec
                        else:
                            random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
                            src_hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]
                            
                            random_spec = int(randint(0, target_spec.shape[0] - self.hint_window, 1))
                            tgt_hint_piece = target_spec[random_spec:random_spec + self.hint_window, :]

                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: piece, 
                                        self.src_hint_placeholder: src_hint_piece,
                                        self.tgt_hint_placeholder: tgt_hint_piece,
                                        self.target_placeholder: target_piece})
            else:
                length = input_waveform.shape[0]
                num_pieces = 1
#                 print(num_pieces)
                indices = randint(0, length-self.sample_size, num_pieces) 
                
#                 if self.spec_generator:
#                     spec_indices = librosa.core.samples_to_frames(indices + int(self.sample_size/2), hop_length=self.spec_generator.shift, n_fft=self.spec_generator.n_fft)
                
                for i in range(num_pieces):
                    idx = indices[i]
                    central = int(idx + self.sample_size/2-self.target_size/2)
                    piece = input_waveform[idx:idx+self.sample_size, :]
                    target_piece = target_waveform[central:central+self.target_size, :]

                    if self.spec_generator:
                        if self.hint_window<=0:
                            src_hint_piece = input_spec
                            tgt_hint_piece = target_spec
                        else:
#                             random_spec = spec_indices[i]
#                             random_shift = randint(-int(self.hint_window/4), int(self.hint_window/4), 1)
#                             random_start = max(0, int(random_spec - self.hint_window/2 + random_shift))
#                             random_end = min(int(random_spec + self.hint_window/2 + random_shift), src_total_frames)
#                             src_hint_piece = input_spec[random_start:random_end, :]
                            
#                             random_shift = randint(-int(self.hint_window/4), int(self.hint_window/4), 1)
#                             random_start = max(0, int(random_spec - self.hint_window/2 + random_shift))
#                             random_end = min(int(random_spec + self.hint_window/2 + random_shift), tgt_total_frames)
#                             tgt_hint_piece = target_spec[random_start:random_end, :]
                            random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
                            src_hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]
        
                            random_spec = int(randint(0, target_spec.shape[0] - self.hint_window, 1))
                            tgt_hint_piece = target_spec[random_spec:random_spec + self.hint_window, :]

                    sess.run(self.enqueue,
                         feed_dict={self.sample_placeholder: piece, 
                                    self.src_hint_placeholder: src_hint_piece,
                                    self.tgt_hint_placeholder: tgt_hint_piece,
                                    self.target_placeholder: target_piece})

    # During testing, use entire audio file for spectrogram frames
    def thread_test(self, sess):
        stop = False
        infer_sample_size = self.target_size * 3 + self.sample_size
        # Go through the dataset multiple times
        mix_generator = MixGeneratorSpec_single(self.test_filenames, self.reverb_train_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=True,
                                          augment_speech=False,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=False,
                                          norm_volume=self.norm_volume,
                                          raw_stft_similarity_score=self.raw_stft_similarity_score,
                                          norm_reverb_peak=False,
                                          SNRdB=[20],
                                          cut_length=infer_sample_size)
        for hint_waveform, target_waveform, input_waveform, tgt_binary_label, tgt_number in mix_generator:
#         for input_waveform, hint_waveform, target_waveform, src_binary_label, tgt_binary_label, src_number, tgt_number in mix_generator:
            if self.coord.should_stop():
                stop = True
                break

#             print("test:", np.argmax(src_binary_label), np.argmax(tgt_binary_label), src_number, tgt_number, input_waveform.shape, target_waveform.shape)
            print("test: (studio)", np.argmax(tgt_binary_label), tgt_number, input_waveform.shape, target_waveform.shape)

#                 target_waveform, keep_indices = trim_silence(target_waveform[:, 0],
#                                                   self.silence_threshold)
#                 target_waveform = target_waveform.reshape(-1, 1)
#                 if target_waveform.size == 0:
#                     print("Warning: {} was ignored as it contains only "
#                           "silence. Consider decreasing trim_silence "
#                           "threshold, or adjust volume of the audio."
#                           .format(id_dict))

#                 input_waveform = input_waveform[keep_indices[0]:keep_indices[-1], :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
                target_spec = self.spec_generator.__preprocess__(hint_waveform[:, 0])
                print("test: from spec generator", input_spec.shape, target_spec.shape)
            elif self.use_label_class and self.hint_size==len(tgt_binary_label): # and self.hint_size==len(src_binary_label)
#                 src_hint_piece = np.reshape(src_binary_label, (1, self.hint_size))
                src_hint_piece = np.zeros_like((1, self.hint_size))
                tgt_hint_piece = np.reshape(tgt_binary_label, (1, self.hint_size))
                print("test: from binary label", src_hint_piece.shape, tgt_hint_piece.shape)
            else:
                src_hint_piece = np.zeros((1, self.hint_size))
                tgt_hint_piece = np.zeros((1, self.hint_size))
                print("test: from dummy zeros", src_hint_piece.shape, tgt_hint_piece.shape)

#             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]

            piece = input_waveform[:infer_sample_size, :]
            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]
            
            if self.spec_generator:
#                     if self.hint_window<=0:
                src_hint_piece = input_spec
                tgt_hint_piece = target_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

            sess.run(self.test_enqueue,
                     feed_dict={self.test_placeholder: piece, 
                                self.test_src_hint_placeholder: src_hint_piece,
                                self.test_tgt_hint_placeholder: tgt_hint_piece,
                                self.test_target_placeholder: target_piece})

    # During testing, use entire audio file for spectrogram frames
    def thread_test_ext(self, sess):
        stop = False
        infer_sample_size = self.target_size * 3 + self.sample_size
        # Go through the dataset multiple times
        mix_generator = MixGeneratorSpec_single(self.test_filenames, self.reverb_test_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=True,
                                          augment_speech=False,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=False,
                                          norm_volume=self.norm_volume,
                                          raw_stft_similarity_score=self.raw_stft_similarity_score,
                                          norm_reverb_peak=False,
                                          SNRdB=[20],
                                          cut_length=infer_sample_size)
        for hint_waveform, target_waveform, input_waveform, tgt_binary_label, tgt_number in mix_generator:
#         for input_waveform, hint_waveform, target_waveform, src_binary_label, tgt_binary_label, src_number, tgt_number in mix_generator:
            if self.coord.should_stop():
                stop = True
                break
#             print("Ext test:", np.argmax(src_binary_label), np.argmax(tgt_binary_label), src_number, tgt_number, input_waveform.shape, target_waveform.shape)
            print("Ext test: (studio) ", np.argmax(tgt_binary_label), tgt_number, input_waveform.shape, target_waveform.shape)
            
            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
                target_spec = self.spec_generator.__preprocess__(hint_waveform[:, 0])
            else:
                src_hint_piece = np.zeros((1, self.hint_size))
                tgt_hint_piece = np.zeros((1, self.hint_size))

#             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]

            piece = input_waveform[:infer_sample_size, :]
            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]
            
            if self.spec_generator:
#                     if self.hint_window<=0:
                src_hint_piece = input_spec
                tgt_hint_piece = target_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

            sess.run(self.test_ext_enqueue,
                     feed_dict={self.test_ext_placeholder: piece, 
                                self.test_ext_src_hint_placeholder: src_hint_piece,
                                self.test_ext_tgt_hint_placeholder: tgt_hint_piece,
                                self.test_ext_target_placeholder: target_piece})

    # During testing, use entire audio file for spectrogram frames
    def thread_test_real(self, sess):
        stop = False
        # Go through the dataset multiple times
        mix_generator = query_joint_yield_pair(gender=self.test_data_range["gender"], num=self.test_data_range["num"], 
                                               script=self.test_data_range["script"], device=None, 
                                               scene=None, directory=self.directory, directory_produced=self.directory, 
                                               exam_ignored=False, 
                                               randomized=True,
                                               sample_rate=self.sample_rate)

        for target_waveform, input_waveform, book in mix_generator:
            if self.coord.should_stop():
                stop = True
                break

            print("Real test:", book, input_waveform.shape, target_waveform.shape)

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
                target_spec = self.spec_generator.__preprocess__(target_waveform[:, 0])
            else:
                src_hint_piece = np.zeros((1, self.hint_size))
                tgt_hint_piece = np.zeros((1, self.hint_size))

            count = 0
            max_count = 10

            random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
            input_waveform = input_waveform[random_start:, :]
            target_waveform = target_waveform[random_start:, :]

            while len(input_waveform) > self.sample_size and count<max_count:
                count = count + 1

                piece = input_waveform[:self.sample_size, :]
                input_waveform = input_waveform[int(self.target_size/2):, :]

                start = int((self.sample_size-self.target_size)/2)
                target_piece = target_waveform[start:start+self.target_size, :]
                target_waveform = target_waveform[int(self.target_size/2):, :]

                if self.spec_generator:
#                     if self.hint_window<=0:
                    src_hint_piece = input_spec
                    tgt_hint_piece = target_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

                sess.run(self.test_real_enqueue,
                         feed_dict={self.test_real_placeholder: piece, 
                                    self.test_real_src_hint_placeholder: src_hint_piece,
                                    self.test_real_tgt_hint_placeholder: tgt_hint_piece,
                                    self.test_real_target_placeholder: target_piece})

#     # During inference, use entire audio file for spectrogram frames
#     def thread_infer(self, sess, start_idx=0, end_idx=271):
#         stop = False
#         # Go through the dataset multiple times
#         full_reverb_filenames=self.reverb_train_filenames+self.reverb_test_filenames
#         my_reverb_filenames=full_reverb_filenames[start_idx:end_idx]
#         mix_generator = MixGeneratorSpec_pair(self.test_filenames, my_reverb_filenames, self.noise_filenames, self.sample_rate,
#                                           speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
#                                           num_classes=200,
#                                           shuffle=False,
#                                           augment_speech=False,
#                                           inject_noise=self.inject_noise,
#                                           augment_reverb=False,
#                                           raw_stft_similarity_score=self.raw_stft_similarity_score)

#         for input_waveform, target_waveform, src_binary_label, tgt_binary_label, src_number, tgt_number in mix_generator:
#             if mix_generator.epoch_index>1:
#                 print("All finished")
#                 stop = True
#                 self.infer_queue.close(cancel_pending_enqueues=False)
#                 break
                
#             if self.coord.should_stop():
#                 stop = True
#                 break
# #                 target_waveform, keep_indices = trim_silence(target_waveform[:, 0],
# #                                                   self.silence_threshold)
# #                 target_waveform = target_waveform.reshape(-1, 1)
# #                 if target_waveform.size == 0:
# #                     print("Warning: {} was ignored as it contains only "
# #                           "silence. Consider decreasing trim_silence "
# #                           "threshold, or adjust volume of the audio."
# #                           .format(id_dict))

# #                 input_waveform = input_waveform[keep_indices[0]:keep_indices[-1], :]

#             if self.spec_generator:
#                 input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
#                 target_spec = self.spec_generator.__preprocess__(target_waveform[:, 0])
#             elif self.use_label_class and self.hint_size==len(src_binary_label) and  self.hint_size==len(tgt_binary_label):
#                 src_hint_piece = np.reshape(src_binary_label, (1, self.hint_size))
#                 tgt_hint_piece = np.reshape(tgt_binary_label, (1, self.hint_size))
#             else:
#                 src_hint_piece = np.zeros((1, self.hint_size))
#                 tgt_hint_piece = np.zeros((1, self.hint_size))

#             count = 0
#             max_count = 10

# #             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
# #             input_waveform = input_waveform[random_start:, :]
# #             target_waveform = target_waveform[random_start:, :]

#             while len(input_waveform) > self.sample_size and count<max_count:
#                 count = count + 1

#                 piece = input_waveform[:self.sample_size, :]
#                 input_waveform = input_waveform[int(self.target_size/2):, :]

#                 start = int(self.sample_size/2-self.target_size/2)
#                 target_piece = target_waveform[start:start+self.target_size, :]
#                 target_waveform = target_waveform[int(self.target_size/2):, :]

#                 if self.spec_generator:
# #                     if self.hint_window<=0:
#                     src_hint_piece = input_spec
#                     tgt_hint_piece = target_spec
# #                     else:
# #                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
# #                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

#                 sess.run(self.infer_enqueue,
#                          feed_dict={self.infer_placeholder: piece, 
#                                     self.infer_src_hint_placeholder: src_hint_piece,
#                                     self.infer_tgt_hint_placeholder: tgt_hint_piece,
#                                     self.infer_target_placeholder: target_piece,
#                                     self.infer_src_class_placeholder: np.array([int(src_number)]),
#                                     self.infer_tgt_class_placeholder: np.array([int(tgt_number)])})

    # During inference, use entire audio file for spectrogram frames
    def thread_infer(self, sess, start_idx=0, end_idx=271):
        stop = False
        infer_sample_size = self.target_size * 11 + self.sample_size
        # Go through the dataset multiple times
        full_reverb_filenames=self.reverb_train_filenames+self.reverb_test_filenames
        my_reverb_filenames=full_reverb_filenames[start_idx:end_idx]

        mix_generator = MixGeneratorSpec_single(self.test_filenames, my_reverb_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=False,
                                          augment_speech=False,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=False,
                                          norm_volume=self.norm_volume,
                                          raw_stft_similarity_score=self.raw_stft_similarity_score,
                                          norm_reverb_peak=False,
                                          SNRdB=[20],
                                          cut_length=infer_sample_size)

        for hint_waveform, target_waveform, input_waveform, tgt_binary_label, tgt_number in mix_generator:
            if mix_generator.epoch_index>1:
                print("All finished")
                stop = True
                self.infer_queue.close(cancel_pending_enqueues=False)
                break
                
#             input_waveform = input_waveform[170000:]
#             target_waveform = target_waveform[170000:]

            if self.coord.should_stop():
                stop = True
                break
#                 target_waveform, keep_indices = trim_silence(target_waveform[:, 0],
#                                                   self.silence_threshold)
#                 target_waveform = target_waveform.reshape(-1, 1)
#                 if target_waveform.size == 0:
#                     print("Warning: {} was ignored as it contains only "
#                           "silence. Consider decreasing trim_silence "
#                           "threshold, or adjust volume of the audio."
#                           .format(id_dict))

#                 input_waveform = input_waveform[keep_indices[0]:keep_indices[-1], :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
                target_spec = self.spec_generator.__preprocess__(hint_waveform[:, 0])
            elif self.use_label_class and self.hint_size==len(src_binary_label) and self.hint_size==len(tgt_binary_label):
                src_hint_piece = np.zeros((1, self.hint_size))
                tgt_hint_piece = np.reshape(tgt_binary_label, (1, self.hint_size))
            else:
                src_hint_piece = np.zeros((1, self.hint_size))
                tgt_hint_piece = np.zeros((1, self.hint_size))

#             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]

            piece = input_waveform[:infer_sample_size, :]
#                 input_waveform = input_waveform[int(self.target_size/2):, :]

            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]

            if self.spec_generator:
#                     if self.hint_window<=0:
                src_hint_piece = input_spec
                tgt_hint_piece = target_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

            sess.run(self.infer_enqueue,
                     feed_dict={self.infer_placeholder: piece, 
                                self.infer_src_hint_placeholder: src_hint_piece,
                                self.infer_tgt_hint_placeholder: tgt_hint_piece,
                                self.infer_target_placeholder: target_piece,
                                self.infer_src_class_placeholder: np.array(["h-01|"+tgt_number.split("|")[1]]),
                                self.infer_tgt_class_placeholder: np.array([tgt_number])})

    def thread_disc(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            mix_generator = MixGeneratorSpec_single(self.test_filenames, self.reverb_train_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                              num_classes=200,
                                              shuffle=True,
                                              augment_speech=False,
                                              inject_noise=self.inject_noise)
            
            for input_waveform, target_waveform, binary_label, _ in mix_generator:
                if self.coord.should_stop():
                    stop = True
                    break
                
                while len(target_waveform) > self.sample_size:
                    target_piece = target_waveform[:self.target_size, :]
                    input_piece = input_waveform[:self.target_size, :]
                    
                    target_waveform = target_waveform[int(self.target_size/2):, :]
                    input_waveform = input_waveform[int(self.target_size/2):, :]
                    
                    sess.run(self.disc_enqueue,
                             feed_dict={self.disc_placeholder: target_piece, 
                                        self.disc_ref_placeholder: input_piece})

class DataReader_2Stage(DataReader):
    '''Generic background audio reader that preprocesses audio files
    and en
    s them into a TensorFlow queue.'''

    def __init__(self,
                 directory,
                 coord,
                 sample_size,
                 hint_size,
                 target_size,
                 sample_rate,
                 random_crop=True,
                 queue_size=32,
                 data_range=CLEAN_DATA_RANGE,
                 test_data_range=CLEAN_TEST_DATA_RANGE,
                 disc_thread_enabled=True,
                 spec_generator=None,
                 use_label_class=False,
                 hint_window=256,
                 inject_noise=False,
                 augment_reverb=False,
                 augment_speech=True,
                 norm_volume=False,
                 stft_similarity=None,
                 norm_reverb_peak=True):
        
        super().__init__(directory,
                         coord,
                         sample_size,
                         hint_size,
                         target_size,
                         sample_rate,
                         random_crop=random_crop,
                         queue_size=queue_size,
                         data_range=data_range,
                         test_data_range=test_data_range,
                         disc_thread_enabled=disc_thread_enabled,
                         spec_generator=spec_generator,
                         use_label_class=use_label_class,
                         hint_window=hint_window,
                         inject_noise=inject_noise,
                         augment_reverb=augment_reverb,
                         augment_speech=augment_speech,
                         norm_volume=norm_volume,
                         stft_similarity=stft_similarity,
                         norm_reverb_peak=norm_reverb_peak)
        
        def construct(hint_size, queue_size):
            sample_ph = tf.placeholder(dtype=tf.float32, shape=None)
            hint_ph = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
            speaker_hint_ph = tf.placeholder(dtype=tf.float32, shape=(None, 20))
            interm_ph = tf.placeholder(dtype=tf.float32, shape=None)
            target_ph = tf.placeholder(dtype=tf.float32, shape=None)

            queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32', 'float32', 'float32', 'float32', 'float32'],
                                         shapes=[(None, 1), (None, hint_size), (None, 20),  (None, 1), (None, 1)])
            enqueue = queue.enqueue([sample_ph, 
                                           hint_ph,
                                           speaker_hint_ph,
                                           interm_ph,
                                           target_ph])
            return sample_ph, hint_ph, speaker_hint_ph, interm_ph, target_ph, queue, enqueue

        ######
        self.sample_placeholder, self.hint_placeholder, self.speaker_hint_placeholder, \
           self.interm_placeholder, self.target_placeholder, self.queue, self.enqueue = construct(hint_size, queue_size)
        
        """For val set"""
        self.test_placeholder, self.test_hint_placeholder, self.test_speaker_hint_placeholder, \
           self.test_interm_placeholder, self.test_target_placeholder, self.test_queue, self.test_enqueue = construct(hint_size, 8)
        
        """For test set"""
        self.test_ext_placeholder, self.test_ext_hint_placeholder, self.test_ext_speaker_hint_placeholder, \
           self.test_ext_interm_placeholder, self.test_ext_target_placeholder, self.test_ext_queue, self.test_ext_enqueue = construct(hint_size, 8)

        """For inference"""
        self.infer_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.infer_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.infer_speaker_hint_placeholders = tf.placeholder(dtype=tf.float32, shape=(None, 20))
        self.infer_interm_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.infer_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.infer_class_placeholder = tf.placeholder(dtype=tf.string, shape=None)
        self.infer_queue = tf.PaddingFIFOQueue(queue_size,
                                               ['float32', 'float32', 'float32', 'float32', 'float32', 'string'],
                                               shapes=[(None, 1), (None, hint_size), (None, 20), (None, 1), (None, 1), (None,)])
        self.infer_enqueue = self.infer_queue.enqueue([self.infer_placeholder, 
                                                       self.infer_hint_placeholder,
                                                       self.infer_speaker_hint_placeholders,
                                                       self.infer_interm_placeholder,
                                                       self.infer_target_placeholder,
                                                       self.infer_class_placeholder])
        """For real set"""
        self.test_real_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_real_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_real_speaker_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 20))
        self.test_real_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_real_queue = tf.PaddingFIFOQueue(8,
                                               ['float32', 'float32', 'float32', 'float32'],
                                               shapes=[(None, 1), (None, hint_size), (None, 20),  (None, 1)])
        self.test_real_enqueue = self.test_real_queue.enqueue([self.test_real_placeholder, 
                                                     self.test_real_hint_placeholder,
                                                     self.test_real_speaker_hint_placeholder,
                                                     self.test_real_target_placeholder])

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        mix_generator = MixGeneratorSpec_single(self.train_filenames, self.reverb_train_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=True,
                                          augment_speech=self.augment_speech,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=self.augment_reverb,
                                          norm_volume=self.norm_volume,
                                          norm_reverb_peak=self.norm_reverb_peak,
                                          cut_length=self.sample_size*5.0)

        print("Loading Data...")
        for input_waveform, interm_waveform, target_waveform, binary_label, speaker_binary_label, _ in mix_generator:
            if self.coord.should_stop():
                stop = True
                break

            # padding
            lag = self.target_size
#             random_start = int(randint(0, lag, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
                total_frames = input_spec.shape[0]
#                 print(total_frames)
            elif self.use_label_class and self.hint_size==len(binary_label):
                hint_piece = np.reshape(binary_label, (1, self.hint_size))
            else:
                hint_piece = np.zeros((1, self.hint_size))
                
            speaker_hint_piece = np.reshape(speaker_binary_label, (1, -1))
            
#                 print(np.argmax(binary_label), input_waveform.shape, target_waveform.shape, input_spec.shape if self.spec_generator else 0)
            
            # input_waveform and target_waveform are now of same length, and with 0-padding in front
            if not self.random_crop:
                while len(input_waveform) > self.sample_size:
                    piece = input_waveform[:self.sample_size, :]
                    input_waveform = input_waveform[lag:, :]

                    start = int((self.sample_size-self.target_size)/2)
                    target_piece = target_waveform[start:start+self.target_size, :]
                    target_waveform = target_waveform[lag:, :]
                    
                    start_2 = int((self.sample_size-self.target_size)/4)
                    interm_piece = interm_waveform[start_2:self.sample_size-start_2, :]
                    interm_waveform = interm_waveform[lag:, :]
                    
                    if self.spec_generator:
                        if self.hint_window<=0:
                            hint_piece = input_spec
                        else:
                            random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
                            hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: piece, 
                                        self.hint_placeholder: hint_piece,
                                        self.speaker_hint_placeholder: speaker_hint_piece,
                                        self.interm_placeholder: interm_piece,
                                        self.target_placeholder: target_piece})
            else:
                length = input_waveform.shape[0]
                num_pieces = 1
#                 print(num_pieces)
                indices = randint(0, length-self.sample_size, num_pieces) 
                
#                 if self.spec_generator:
#                     spec_indices = librosa.core.samples_to_frames(indices + int(self.sample_size/2), hop_length=self.spec_generator.shift, n_fft=self.spec_generator.n_fft)
                
                for i in range(num_pieces):
                    idx = indices[i]
                    central = int(idx + (self.sample_size-self.target_size)/2)
                    piece = input_waveform[idx:idx+self.sample_size, :]
                    target_piece = target_waveform[central:central+self.target_size, :]
                    start_2 = int((self.sample_size-self.target_size)/4)
                    interm_piece = interm_waveform[int(idx+start_2):int(idx+self.sample_size-start_2), :]
                    
                    if self.spec_generator:
                        if self.hint_window<=0:
                            hint_piece = input_spec
                        else:
#                             random_spec = spec_indices[i]
#                             random_shift = randint(-int(self.hint_window/4), int(self.hint_window/4), 1)
#                             random_start = max(0, int(random_spec - self.hint_window/2 + random_shift))
#                             random_end = min(int(random_spec + self.hint_window/2 + random_shift), total_frames)
#                             hint_piece = input_spec[random_start:random_end, :]
                            random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
                            hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]
                    
#                     print("train:", np.argmax(binary_label), piece.shape, hint_piece.shape, speaker_hint_piece.shape, interm_piece.shape, target_piece.shape)
                
                    sess.run(self.enqueue,
                         feed_dict={self.sample_placeholder: piece, 
                                    self.hint_placeholder: hint_piece,
                                    self.speaker_hint_placeholder: speaker_hint_piece,
                                    self.interm_placeholder: interm_piece,
                                    self.target_placeholder: target_piece})

    # During testing, use entire audio file for spectrogram frames
    def thread_test(self, sess):
        stop = False
        infer_sample_size = self.target_size * 3 + self.sample_size
        # Go through the dataset multiple times
        mix_generator = MixGeneratorSpec_single(self.test_filenames, self.reverb_train_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=True,
                                          augment_speech=False,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=False,
                                          norm_volume=self.norm_volume,
                                          raw_stft_similarity_score=self.raw_stft_similarity_score,
                                          norm_reverb_peak=False,
                                          SNRdB=[20],
                                          cut_length=infer_sample_size)

        for input_waveform, interm_waveform, target_waveform, binary_label, speaker_binary_label, number in mix_generator:
            if self.coord.should_stop():
                stop = True
                break
                
            print("test:", np.argmax(binary_label), number, input_waveform.shape, target_waveform.shape)
#                 target_waveform, keep_indices = trim_silence(target_waveform[:, 0],
#                                                   self.silence_threshold)
#                 target_waveform = target_waveform.reshape(-1, 1)
#                 if target_waveform.size == 0:
#                     print("Warning: {} was ignored as it contains only "
#                           "silence. Consider decreasing trim_silence "
#                           "threshold, or adjust volume of the audio."
#                           .format(id_dict))

#                 input_waveform = input_waveform[keep_indices[0]:keep_indices[-1], :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
#                 print("test: from spec generator", input_spec.shape)
            elif self.use_label_class and self.hint_size==len(binary_label):
                hint_piece = np.reshape(binary_label, (1, self.hint_size))
#                 print("test: from binary label", hint_piece.shape)
            else:
                hint_piece = np.zeros((1, self.hint_size))
#                 print("test: from dummy zeros", hint_piece.shape)
                
            speaker_hint_piece = np.reshape(speaker_binary_label, (1, -1))

#             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]

            piece = input_waveform[:infer_sample_size, :]
            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]
            start_2 = int((self.sample_size-self.target_size)/4)
            interm_piece = interm_waveform[start_2:infer_sample_size-start_2, :]
            
            if self.spec_generator:
#                     if self.hint_window<=0:
                hint_piece = input_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

            sess.run(self.test_enqueue,
                     feed_dict={self.test_placeholder: piece, 
                                self.test_hint_placeholder: hint_piece,
                                self.test_speaker_hint_placeholder: speaker_hint_piece,
                                self.test_interm_placeholder: interm_piece,
                                self.test_target_placeholder: target_piece})
    
    # During testing, use entire audio file for spectrogram frames
    def thread_test_ext(self, sess):
        stop = False
        infer_sample_size = self.target_size * 3 + self.sample_size
        # Go through the dataset multiple times
        mix_generator = MixGeneratorSpec_single(self.test_filenames, self.reverb_test_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=True,
                                          augment_speech=False,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=False,
                                          norm_volume=self.norm_volume,
                                          raw_stft_similarity_score=self.raw_stft_similarity_score,
                                          norm_reverb_peak=False,
                                          SNRdB=[20],
                                          cut_length=infer_sample_size)

        for input_waveform, interm_waveform, target_waveform, binary_label, speaker_binary_label, number in mix_generator:
            if self.coord.should_stop():
                stop = True
                break

            print("Ext test:", np.argmax(binary_label), number, input_waveform.shape, target_waveform.shape)

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
            else:
                hint_piece = np.zeros((1, self.hint_size))
                
            speaker_hint_piece = np.reshape(speaker_binary_label, (1, -1))

#             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]
            piece = input_waveform[:infer_sample_size, :]
            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]
            start_2 = int((self.sample_size-self.target_size)/4)
            interm_piece = interm_waveform[start_2:infer_sample_size-start_2, :]
            
            if self.spec_generator:
#                     if self.hint_window<=0:
                hint_piece = input_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

            sess.run(self.test_ext_enqueue,
                     feed_dict={self.test_ext_placeholder: piece, 
                                self.test_ext_hint_placeholder: hint_piece,
                                self.test_ext_speaker_hint_placeholder: speaker_hint_piece,
                                self.test_ext_interm_placeholder: interm_piece,
                                self.test_ext_target_placeholder: target_piece})

    # During testing, use entire audio file for spectrogram frames
    def thread_test_real(self, sess):
        stop = False
        # Go through the dataset multiple times
        mix_generator = query_joint_yield_pair(gender=self.test_data_range["gender"], num=self.test_data_range["num"], 
                                               script=self.test_data_range["script"], device=None, 
                                               scene=None, directory=self.directory, directory_produced=self.directory, 
                                               exam_ignored=False, 
                                               randomized=True,
                                               sample_rate=self.sample_rate)

        for input_waveform, target_waveform, book in mix_generator:
            if self.coord.should_stop():
                stop = True
                break

            print("Real test:", book, input_waveform.shape, target_waveform.shape)

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
            else:
                hint_piece = np.zeros((1, self.hint_size))
            
            speaker_id, speaker_binary_label = extract_speaker_id(book["gender"]+str(book["num"]))
            speaker_hint_piece = np.reshape(speaker_binary_label, (1, -1))

            count = 0
            max_count = 10

            random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
            input_waveform = input_waveform[random_start:, :]
            target_waveform = target_waveform[random_start:, :]

            while len(input_waveform) > self.sample_size and count<max_count:
                count = count + 1

                piece = input_waveform[:self.sample_size, :]
                input_waveform = input_waveform[int(self.target_size/2):, :]

                start = int((self.sample_size-self.target_size)/2)
                target_piece = target_waveform[start:start+self.target_size, :]
                target_waveform = target_waveform[int(self.target_size/2):, :]

                if self.spec_generator:
#                     if self.hint_window<=0:
                    hint_piece = input_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

                sess.run(self.test_real_enqueue,
                         feed_dict={self.test_real_placeholder: piece, 
                                    self.test_real_hint_placeholder: hint_piece,
                                    self.test_real_speaker_hint_placeholder: speaker_hint_piece,
                                    self.test_real_target_placeholder: target_piece})

    # During inference, use entire audio file for spectrogram frames
    def thread_infer(self, sess, start_idx=0, end_idx=271):
        stop = False
        infer_sample_size = self.target_size * 11 + self.sample_size
        # Go through the dataset multiple times
        full_reverb_filenames=self.reverb_train_filenames+self.reverb_test_filenames
        my_reverb_filenames=full_reverb_filenames[start_idx:end_idx]
        mix_generator = MixGeneratorSpec_single(self.test_filenames, my_reverb_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=False,
                                          augment_speech=False,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=False,
                                          norm_volume=self.norm_volume,
                                          raw_stft_similarity_score=self.raw_stft_similarity_score,
                                          norm_reverb_peak=False,
                                          SNRdB=[20],
                                          cut_length=infer_sample_size)

        for input_waveform, interm_waveform, target_waveform, binary_label, speaker_binary_label, number in mix_generator:
            if mix_generator.epoch_index>1:
                print("All finished")
                stop = True
                self.infer_queue.close(cancel_pending_enqueues=False)
                break
                
#             input_waveform = input_waveform[170000:]
#             target_waveform = target_waveform[170000:]

            if self.coord.should_stop():
                stop = True
                break
#                 target_waveform, keep_indices = trim_silence(target_waveform[:, 0],
#                                                   self.silence_threshold)
#                 target_waveform = target_waveform.reshape(-1, 1)
#                 if target_waveform.size == 0:
#                     print("Warning: {} was ignored as it contains only "
#                           "silence. Consider decreasing trim_silence "
#                           "threshold, or adjust volume of the audio."
#                           .format(id_dict))

#                 input_waveform = input_waveform[keep_indices[0]:keep_indices[-1], :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
            elif self.use_label_class and self.hint_size==len(binary_label):
                hint_piece = np.reshape(binary_label, (1, self.hint_size))
            else:
                hint_piece = np.zeros((1, self.hint_size))
                
            speaker_hint_piece = np.reshape(speaker_binary_label, (1, -1))

#             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]

            piece = input_waveform[:infer_sample_size, :]

            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]
            start_2 = int((self.sample_size-self.target_size)/4)
            interm_piece = interm_waveform[start_2:infer_sample_size-start_2, :]
            
            if self.spec_generator:
#                     if self.hint_window<=0:
                hint_piece = input_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

            sess.run(self.infer_enqueue,
                     feed_dict={self.infer_placeholder: piece, 
                                self.infer_hint_placeholder: hint_piece,
                                self.infer_speaker_hint_placeholder: speaker_hint_piece,
                                self.infer_interm_placeholder: interm_piece,
                                self.infer_target_placeholder: target_piece,
                                self.infer_class_placeholder: np.array([number])})

class DataReader_pair_multi(DataReader):
    '''Generic background audio reader that preprocesses audio files
    and en
    s them into a TensorFlow queue.'''

    def __init__(self,
                 directory,
                 coord,
                 sample_size,
                 hint_size,
                 target_size,
                 sample_rate,
                 random_crop=True,
                 queue_size=32,
                 data_range=CLEAN_DATA_RANGE,
                 test_data_range=CLEAN_TEST_DATA_RANGE,
                 disc_thread_enabled=True,
                 spec_generator=None,
                 use_label_class=False,
                 hint_window=256,
                 inject_noise=False,
                 augment_reverb=False,
                 augment_speech=True,
                 norm_volume=False,
                 stft_similarity=None,
                 norm_reverb_peak=True):
        
        super().__init__(directory,
                         coord,
                         sample_size,
                         hint_size,
                         target_size,
                         sample_rate,
                         random_crop=random_crop,
                         queue_size=queue_size,
                         data_range=data_range,
                         test_data_range=test_data_range,
                         disc_thread_enabled=disc_thread_enabled,
                         spec_generator=spec_generator,
                         use_label_class=use_label_class,
                         hint_window=hint_window,
                         inject_noise=inject_noise,
                         augment_reverb=augment_reverb,
                         augment_speech=augment_speech,
                         norm_volume=norm_volume,
                         stft_similarity=stft_similarity,
                         norm_reverb_peak=norm_reverb_peak)
        
        ######
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.src_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.tgt_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.clean_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)

        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32', 'float32', 'float32', 'float32', 'float32'],
                                         shapes=[(None, 1), (None, hint_size), (None, hint_size),  (None, 1), (None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder, 
                                           self.src_hint_placeholder,
                                           self.tgt_hint_placeholder,
                                           self.target_placeholder,
                                           self.clean_target_placeholder])

        """For val set"""
        self.test_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_src_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_tgt_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_clean_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_queue = tf.PaddingFIFOQueue(8,
                                              ['float32', 'float32', 'float32', 'float32', 'float32'],
                                              shapes=[(None, 1), (None, hint_size), (None, hint_size),  (None, 1), (None, 1)])
        self.test_enqueue = self.test_queue.enqueue([self.test_placeholder, 
                                                     self.test_src_hint_placeholder,
                                                     self.test_tgt_hint_placeholder,
                                                     self.test_target_placeholder,
                                                     self.test_clean_target_placeholder])
        """For test set"""
        self.test_ext_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_ext_src_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_ext_tgt_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_ext_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_ext_clean_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_ext_queue = tf.PaddingFIFOQueue(8,
                                                  ['float32', 'float32', 'float32', 'float32', 'float32'],
                                                  shapes=[(None, 1), (None, hint_size), (None, hint_size),  (None, 1), (None, 1)])
        self.test_ext_enqueue = self.test_ext_queue.enqueue([self.test_ext_placeholder, 
                                                     self.test_ext_src_hint_placeholder,
                                                     self.test_ext_tgt_hint_placeholder,
                                                     self.test_ext_target_placeholder,
                                                     self.test_ext_clean_target_placeholder])
        """For inference"""
        self.infer_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.infer_src_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.infer_tgt_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.infer_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.infer_clean_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.infer_src_class_placeholder = tf.placeholder(dtype=tf.string, shape=None)
        self.infer_tgt_class_placeholder = tf.placeholder(dtype=tf.string, shape=None)
        self.infer_queue = tf.PaddingFIFOQueue(queue_size,
                                               ['float32', 'float32', 'float32', 'float32', 'float32', 'string', 'string'],
                                               shapes=[(None, 1), (None, hint_size), (None, hint_size),  (None, 1), (None, 1), (None,), (None,)])
        self.infer_enqueue = self.infer_queue.enqueue([self.infer_placeholder, 
                                                       self.infer_src_hint_placeholder,
                                                       self.infer_tgt_hint_placeholder,
                                                       self.infer_target_placeholder,
                                                       self.infer_clean_target_placeholder,
                                                       self.infer_src_class_placeholder,
                                                       self.infer_tgt_class_placeholder])
        """For real set"""
        self.test_real_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_real_src_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_real_tgt_hint_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, hint_size))
        self.test_real_target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.test_real_queue = tf.PaddingFIFOQueue(8,
                                               ['float32', 'float32', 'float32', 'float32'],
                                               shapes=[(None, 1), (None, hint_size), (None, hint_size),  (None, 1)])
        self.test_real_enqueue = self.test_real_queue.enqueue([self.test_real_placeholder, 
                                                     self.test_real_src_hint_placeholder,
                                                     self.test_real_tgt_hint_placeholder,
                                                     self.test_real_target_placeholder])

    # During training, select spectrogram frames centered around waveform piece if random_crop enbaled; otherwise randomly
    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        mix_generator = MixGeneratorSpec_pair(self.train_filenames, self.reverb_train_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=True,
                                          augment_speech=self.augment_speech,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=self.augment_reverb,
                                          norm_volume=self.norm_volume,
                                          norm_reverb_peak=self.norm_reverb_peak,
                                          cut_length=self.sample_size*5.0)

        print("Loading Data...")
        for input_waveform, hint_waveform, target_waveform, clean_waveform, src_binary_label, tgt_binary_label, _, _ in mix_generator:
            if self.coord.should_stop():
                stop = True
                break

            # padding
            lag = self.target_size
            random_start = int(randint(0, lag, 1))
            input_waveform = input_waveform[random_start:, :]
            target_waveform = target_waveform[random_start:, :]
            clean_waveform = clean_waveform[random_start:, :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
                src_total_frames = input_spec.shape[0]
                
                target_spec = self.spec_generator.__preprocess__(hint_waveform[:, 0])
                tgt_total_frames = target_spec.shape[0]
            elif self.use_label_class and self.hint_size==len(src_binary_label) and self.hint_size==len(tgt_binary_label):
                src_hint_piece = np.reshape(src_binary_label, (1, self.hint_size))
                tgt_hint_piece = np.reshape(tgt_binary_label, (1, self.hint_size))
            else:
                src_hint_piece = np.zeros((1, self.hint_size))
                tgt_hint_piece = np.zeros((1, self.hint_size))

#                 print(np.argmax(binary_label), input_waveform.shape, target_waveform.shape, input_spec.shape if self.spec_generator else 0)

            # input_waveform and target_waveform are now of same length, and with 0-padding in front
            if not self.random_crop:
                while len(input_waveform) > self.sample_size:
                    piece = input_waveform[:self.sample_size, :]
                    input_waveform = input_waveform[lag:, :]

                    start = int((self.sample_size-self.target_size)/2)
                    target_piece = target_waveform[start:start+self.target_size, :]
                    target_waveform = target_waveform[lag:, :]
                    start_2 = int((self.sample_size-self.target_size)/4)
                    clean_piece = clean_waveform[start_2:self.sample_size-start_2, :]
                    clean_waveform = clean_waveform[lag:, :]

                    if self.spec_generator:
                        if self.hint_window<=0:
                            src_hint_piece = input_spec
                            tgt_hint_piece = target_spec
                        else:
                            random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
                            src_hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]
                            
                            random_spec = int(randint(0, target_spec.shape[0] - self.hint_window, 1))
                            tgt_hint_piece = target_spec[random_spec:random_spec + self.hint_window, :]

                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: piece, 
                                        self.src_hint_placeholder: src_hint_piece,
                                        self.tgt_hint_placeholder: tgt_hint_piece,
                                        self.target_placeholder: target_piece,
                                        self.clean_target_placeholder: clean_piece})
            else:
                length = input_waveform.shape[0]
                num_pieces = 1
#                 print(num_pieces)
                indices = randint(0, length-self.sample_size, num_pieces) 
                
#                 if self.spec_generator:
#                     spec_indices = librosa.core.samples_to_frames(indices + int(self.sample_size/2), hop_length=self.spec_generator.shift, n_fft=self.spec_generator.n_fft)
                
                for i in range(num_pieces):
                    idx = indices[i]
                    central = int(idx + self.sample_size/2-self.target_size/2)
                    piece = input_waveform[idx:idx+self.sample_size, :]
                    target_piece = target_waveform[central:central+self.target_size, :]
                    start_2 = int((self.sample_size-self.target_size)/4)
                    clean_piece = clean_waveform[int(idx+start_2):int(idx+self.sample_size-start_2), :]
                    
                    if self.spec_generator:
                        if self.hint_window<=0:
                            src_hint_piece = input_spec
                            tgt_hint_piece = target_spec
                        else:
#                             random_spec = spec_indices[i]
#                             random_shift = randint(-int(self.hint_window/4), int(self.hint_window/4), 1)
#                             random_start = max(0, int(random_spec - self.hint_window/2 + random_shift))
#                             random_end = min(int(random_spec + self.hint_window/2 + random_shift), src_total_frames)
#                             src_hint_piece = input_spec[random_start:random_end, :]
                            
#                             random_shift = randint(-int(self.hint_window/4), int(self.hint_window/4), 1)
#                             random_start = max(0, int(random_spec - self.hint_window/2 + random_shift))
#                             random_end = min(int(random_spec + self.hint_window/2 + random_shift), tgt_total_frames)
#                             tgt_hint_piece = target_spec[random_start:random_end, :]
                            random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
                            src_hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]
        
                            random_spec = int(randint(0, target_spec.shape[0] - self.hint_window, 1))
                            tgt_hint_piece = target_spec[random_spec:random_spec + self.hint_window, :]

                    sess.run(self.enqueue,
                         feed_dict={self.sample_placeholder: piece, 
                                    self.src_hint_placeholder: src_hint_piece,
                                    self.tgt_hint_placeholder: tgt_hint_piece,
                                    self.target_placeholder: target_piece,
                                    self.clean_target_placeholder: clean_piece})

    # During testing, use entire audio file for spectrogram frames
    def thread_test(self, sess):
        stop = False
        infer_sample_size = self.target_size * 4 + self.sample_size
        # Go through the dataset multiple times
        mix_generator = MixGeneratorSpec_single(self.test_filenames, self.reverb_train_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=True,
                                          augment_speech=False,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=False,
                                          norm_volume=self.norm_volume,
                                          raw_stft_similarity_score=self.raw_stft_similarity_score,
                                          norm_reverb_peak=False,
                                          cut_length=infer_sample_size)
        for hint_waveform, target_waveform, input_waveform, tgt_binary_label, tgt_number in mix_generator:
#         for input_waveform, hint_waveform, target_waveform, src_binary_label, tgt_binary_label, src_number, tgt_number in mix_generator:
            if self.coord.should_stop():
                stop = True
                break

#             print("test:", np.argmax(src_binary_label), np.argmax(tgt_binary_label), src_number, tgt_number, input_waveform.shape, target_waveform.shape)
            print("test: (studio)", np.argmax(tgt_binary_label), tgt_number, input_waveform.shape, target_waveform.shape)

#                 target_waveform, keep_indices = trim_silence(target_waveform[:, 0],
#                                                   self.silence_threshold)
#                 target_waveform = target_waveform.reshape(-1, 1)
#                 if target_waveform.size == 0:
#                     print("Warning: {} was ignored as it contains only "
#                           "silence. Consider decreasing trim_silence "
#                           "threshold, or adjust volume of the audio."
#                           .format(id_dict))

#                 input_waveform = input_waveform[keep_indices[0]:keep_indices[-1], :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
                target_spec = self.spec_generator.__preprocess__(hint_waveform[:, 0])
                print("test: from spec generator", input_spec.shape, target_spec.shape)
            elif self.use_label_class and self.hint_size==len(tgt_binary_label): # and self.hint_size==len(src_binary_label)
#                 src_hint_piece = np.reshape(src_binary_label, (1, self.hint_size))
                src_hint_piece = np.zeros_like((1, self.hint_size))
                tgt_hint_piece = np.reshape(tgt_binary_label, (1, self.hint_size))
                print("test: from binary label", src_hint_piece.shape, tgt_hint_piece.shape)
            else:
                src_hint_piece = np.zeros((1, self.hint_size))
                tgt_hint_piece = np.zeros((1, self.hint_size))
                print("test: from dummy zeros", src_hint_piece.shape, tgt_hint_piece.shape)

#             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]


            piece = input_waveform[:infer_sample_size, :]
            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]
            start_2 = int((self.sample_size-self.target_size)/4)
            clean_piece = piece[start_2:-start_2, :]

            if self.spec_generator:
#                     if self.hint_window<=0:
                src_hint_piece = input_spec
                tgt_hint_piece = target_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

            sess.run(self.test_enqueue,
                     feed_dict={self.test_placeholder: piece, 
                                self.test_src_hint_placeholder: src_hint_piece,
                                self.test_tgt_hint_placeholder: tgt_hint_piece,
                                self.test_target_placeholder: target_piece,
                                self.test_clean_target_placeholder: clean_piece})

    # During testing, use entire audio file for spectrogram frames
    def thread_test_ext(self, sess):
        stop = False
        infer_sample_size = self.target_size * 4 + self.sample_size
        # Go through the dataset multiple times
        mix_generator = MixGeneratorSpec_single(self.test_filenames, self.reverb_test_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=True,
                                          augment_speech=False,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=False,
                                          norm_volume=self.norm_volume,
                                          raw_stft_similarity_score=self.raw_stft_similarity_score,
                                          norm_reverb_peak=False,
                                          cut_length=infer_sample_size)
        for hint_waveform, target_waveform, input_waveform, tgt_binary_label, tgt_number in mix_generator:
#         for input_waveform, hint_waveform, target_waveform, src_binary_label, tgt_binary_label, src_number, tgt_number in mix_generator:
            if self.coord.should_stop():
                stop = True
                break
#             print("Ext test:", np.argmax(src_binary_label), np.argmax(tgt_binary_label), src_number, tgt_number, input_waveform.shape, target_waveform.shape)
            print("Ext test: (studio) ", np.argmax(tgt_binary_label), tgt_number, input_waveform.shape, target_waveform.shape)
            
            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
                target_spec = self.spec_generator.__preprocess__(hint_waveform[:, 0])
            else:
                src_hint_piece = np.zeros((1, self.hint_size))
                tgt_hint_piece = np.zeros((1, self.hint_size))
            
#             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]

            piece = input_waveform[:infer_sample_size, :]
            
            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]
            start_2 = int((self.sample_size-self.target_size)/4)
            clean_piece = piece[start_2:-start_2, :]

            if self.spec_generator:
#                     if self.hint_window<=0:
                src_hint_piece = input_spec
                tgt_hint_piece = target_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

            sess.run(self.test_ext_enqueue,
                     feed_dict={self.test_ext_placeholder: piece, 
                                self.test_ext_src_hint_placeholder: src_hint_piece,
                                self.test_ext_tgt_hint_placeholder: tgt_hint_piece,
                                self.test_ext_target_placeholder: target_piece,
                                self.test_ext_clean_target_placeholder: clean_piece})

#     # During inference, use entire audio file for spectrogram frames
#     def thread_infer(self, sess, start_idx=0, end_idx=271):
#         stop = False
#         # Go through the dataset multiple times
#         full_reverb_filenames=self.reverb_train_filenames+self.reverb_test_filenames
#         my_reverb_filenames=full_reverb_filenames[start_idx:end_idx]
#         mix_generator = MixGeneratorSpec_pair(self.test_filenames, my_reverb_filenames, self.noise_filenames,
#                                           speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
#                                           sample_rate=self.sample_rate,
#                                           num_classes=200,
#                                           shuffle=False,
#                                           augment_speech=False,
#                                           inject_noise=self.inject_noise,
#                                           augment_reverb=False,
#                                           raw_stft_similarity_score=self.raw_stft_similarity_score)

#         for input_waveform, target_waveform, src_binary_label, tgt_binary_label, src_number, tgt_number in mix_generator:
#             if mix_generator.epoch_index>1:
#                 print("All finished")
#                 stop = True
#                 self.infer_queue.close(cancel_pending_enqueues=False)
#                 break
                
#             if self.coord.should_stop():
#                 stop = True
#                 break
# #                 target_waveform, keep_indices = trim_silence(target_waveform[:, 0],
# #                                                   self.silence_threshold)
# #                 target_waveform = target_waveform.reshape(-1, 1)
# #                 if target_waveform.size == 0:
# #                     print("Warning: {} was ignored as it contains only "
# #                           "silence. Consider decreasing trim_silence "
# #                           "threshold, or adjust volume of the audio."
# #                           .format(id_dict))

# #                 input_waveform = input_waveform[keep_indices[0]:keep_indices[-1], :]

#             if self.spec_generator:
#                 input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
#                 target_spec = self.spec_generator.__preprocess__(target_waveform[:, 0])
#             elif self.use_label_class and self.hint_size==len(src_binary_label) and  self.hint_size==len(tgt_binary_label):
#                 src_hint_piece = np.reshape(src_binary_label, (1, self.hint_size))
#                 tgt_hint_piece = np.reshape(tgt_binary_label, (1, self.hint_size))
#             else:
#                 src_hint_piece = np.zeros((1, self.hint_size))
#                 tgt_hint_piece = np.zeros((1, self.hint_size))

#             count = 0
#             max_count = 10

# #             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
# #             input_waveform = input_waveform[random_start:, :]
# #             target_waveform = target_waveform[random_start:, :]

#             while len(input_waveform) > self.sample_size and count<max_count:
#                 count = count + 1

#                 piece = input_waveform[:self.sample_size, :]
#                 input_waveform = input_waveform[int(self.target_size/2):, :]

#                 start = int(self.sample_size/2-self.target_size/2)
#                 target_piece = target_waveform[start:start+self.target_size, :]
#                 target_waveform = target_waveform[int(self.target_size/2):, :]

#                 if self.spec_generator:
# #                     if self.hint_window<=0:
#                     src_hint_piece = input_spec
#                     tgt_hint_piece = target_spec
# #                     else:
# #                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
# #                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

#                 sess.run(self.infer_enqueue,
#                          feed_dict={self.infer_placeholder: piece, 
#                                     self.infer_src_hint_placeholder: src_hint_piece,
#                                     self.infer_tgt_hint_placeholder: tgt_hint_piece,
#                                     self.infer_target_placeholder: target_piece,
#                                     self.infer_src_class_placeholder: np.array([int(src_number)]),
#                                     self.infer_tgt_class_placeholder: np.array([int(tgt_number)])})

    # During inference, use entire audio file for spectrogram frames
    def thread_infer(self, sess, start_idx=0, end_idx=271):
        stop = False
        infer_sample_size = self.target_size * 11 + self.sample_size
        # Go through the dataset multiple times
        full_reverb_filenames=self.reverb_train_filenames+self.reverb_test_filenames
        my_reverb_filenames=full_reverb_filenames[start_idx:end_idx]

        mix_generator = MixGeneratorSpec_single(self.test_filenames, my_reverb_filenames, self.noise_filenames, self.sample_rate,
                                          speech_data_holder=None, reverb_data_holder=None, noise_data_holder=None,
                                          num_classes=200,
                                          shuffle=False,
                                          augment_speech=False,
                                          inject_noise=self.inject_noise,
                                          augment_reverb=False,
                                          norm_volume=self.norm_volume,
                                          raw_stft_similarity_score=self.raw_stft_similarity_score,
                                          norm_reverb_peak=False,
                                          cut_length=infer_sample_size)
        
        for hint_waveform, target_waveform, input_waveform, tgt_binary_label, tgt_number in mix_generator:
            if mix_generator.epoch_index>1:
                print("All finished")
                stop = True
                self.infer_queue.close(cancel_pending_enqueues=False)
                break
                
#             input_waveform = input_waveform[170000:]
#             target_waveform = target_waveform[170000:]

            if self.coord.should_stop():
                stop = True
                break
#                 target_waveform, keep_indices = trim_silence(target_waveform[:, 0],
#                                                   self.silence_threshold)
#                 target_waveform = target_waveform.reshape(-1, 1)
#                 if target_waveform.size == 0:
#                     print("Warning: {} was ignored as it contains only "
#                           "silence. Consider decreasing trim_silence "
#                           "threshold, or adjust volume of the audio."
#                           .format(id_dict))

#                 input_waveform = input_waveform[keep_indices[0]:keep_indices[-1], :]

            if self.spec_generator:
                input_spec = self.spec_generator.__preprocess__(input_waveform[:, 0])
                target_spec = self.spec_generator.__preprocess__(hint_waveform[:, 0])
            elif self.use_label_class and self.hint_size==len(src_binary_label) and  self.hint_size==len(tgt_binary_label):
                src_hint_piece = np.zeros((1, self.hint_size))
                tgt_hint_piece = np.reshape(tgt_binary_label, (1, self.hint_size))
            else:
                src_hint_piece = np.zeros((1, self.hint_size))
                tgt_hint_piece = np.zeros((1, self.hint_size))

#             random_start = int(randint(0, input_waveform.shape[0]-max_count*self.sample_size, 1))
#             input_waveform = input_waveform[random_start:, :]
#             target_waveform = target_waveform[random_start:, :]

            piece = input_waveform[:infer_sample_size, :]
#                 input_waveform = input_waveform[int(self.target_size/2):, :]

            start = int((self.sample_size-self.target_size)/2)
            target_piece = target_waveform[start:infer_sample_size-start, :]
            start_2 = int((self.sample_size-self.target_size)/4)
            clean_piece = piece[start_2:-start_2, :]
            
            if self.spec_generator:
#                     if self.hint_window<=0:
                src_hint_piece = input_spec
                tgt_hint_piece = target_spec
#                     else:
#                         random_spec = int(randint(0, input_spec.shape[0] - self.hint_window, 1))
#                         hint_piece = input_spec[random_spec:random_spec + self.hint_window, :]

            sess.run(self.infer_enqueue,
                     feed_dict={self.infer_placeholder: piece, 
                                self.infer_src_hint_placeholder: src_hint_piece,
                                self.infer_tgt_hint_placeholder: tgt_hint_piece,
                                self.infer_target_placeholder: target_piece,
                                self.infer_clean_target_placeholder: clean_target_piece,
                                self.infer_src_class_placeholder: np.array(["h-01|"+tgt_number.split("|")[1]]),
                                self.infer_tgt_class_placeholder: np.array([tgt_number])})



DATAREADER_FACTORY = {"DataReader": DataReader, "DataReader_pair": DataReader_pair, "DataReader_pair_multi": DataReader_pair_multi}