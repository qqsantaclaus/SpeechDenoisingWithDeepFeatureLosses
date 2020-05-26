from model import *
from data_import import *
from data_reader import DataGenerator
from speech_embedding.emb_data_generator import query_joint_yield, CLEAN_DATA_RANGE, CLEAN_TEST_DATA_RANGE
from speech_embedding.read_Audio_RIRs import read_Audio_RIRs, get_Audio_RIR_classes, read_noise

import sys, getopt

# SPEECH ENHANCEMENT NETWORK
SE_LAYERS = 13 # NUMBER OF INTERNAL LAYERS
SE_CHANNELS = 64 # NUMBER OF FEATURE CHANNELS PER LAYER
SE_LOSS_LAYERS = 6 # NUMBER OF FEATURE LOSS LAYERS
SE_NORM = "NM" # TYPE OF LAYER NORMALIZATION (NM, SBN or None)
SE_LOSS_TYPE = "FL" # TYPE OF TRAINING LOSS (L1, L2 or FL)

# FEATURE LOSS NETWORK
LOSS_LAYERS = 14 # NUMBER OF INTERNAL LAYERS
LOSS_BASE_CHANNELS = 32 # NUMBER OF FEATURE CHANNELS PER LAYER IN FIRT LAYER
LOSS_BLK_CHANNELS = 5 # NUMBER OF LAYERS BETWEEN CHANNEL NUMBER UPDATES
LOSS_NORM = "SBN" # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

SET_WEIGHT_EPOCH = 10 # NUMBER OF EPOCHS BEFORE FEATURE LOSS BALANCE
SAVE_EPOCHS = 10 # NUMBER OF EPOCHS BETWEEN MODEL SAVES

log_file = open("logfile.txt", 'w+')

# COMMAND LINE OPTIONS
datafolder = "dataset"
modfolder = "models"
outfolder = "."
restore_from = "models_sim_320/se_model.ckpt"
try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:l:o:",["ifolder=,lossfolder=,outfolder="])
except getopt.GetoptError:
    print('Usage: python senet_infer.py -d <datafolder> -l <lossfolder> -o <outfolder>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('Usage: python senet_infer.py -d <datafolder> -l <lossfolder> -o <outfolder>')
        sys.exit()
    elif opt in ("-d", "--datafolder"):
        datafolder = arg
    elif opt in ("-l", "--lossfolder"):
        modfolder = arg
    elif opt in ("-o", "--outfolder"):
        outfolder = arg
print('Data folder is "' + datafolder + '/"')
print('Loss model folder is "' + modfolder + '/"')
print('Restore model from "' + restore_from + '/"')
print('Output model folder is "' + outfolder + '/"')

# SET LOSS FUNCTIONS AND PLACEHOLDERS
with tf.variable_scope(tf.get_variable_scope()):
    input=tf.placeholder(tf.float32,shape=[None,1,None,1])
    clean=tf.placeholder(tf.float32,shape=[None,1,None,1])
        
    enhanced=senet(input, n_layers=SE_LAYERS, norm_type=SE_NORM, n_channels=SE_CHANNELS)
        
    if SE_LOSS_TYPE == "L1": # L1 LOSS
        loss_weights = tf.placeholder(tf.float32, shape=[])
        loss_fn = l1_loss(clean, enhanced)
    elif SE_LOSS_TYPE == "L2": # L2 LOSS
        loss_weights = tf.placeholder(tf.float32, shape=[])
        loss_fn = l2_loss(clean, enhanced)
    else: # FEATURE LOSS
        loss_weights = tf.placeholder(tf.float32, shape=[SE_LOSS_LAYERS])
        loss_fn = featureloss(clean, enhanced, loss_weights, loss_layers=SE_LOSS_LAYERS, n_layers=LOSS_LAYERS, norm_type=LOSS_NORM,
                                 base_channels=LOSS_BASE_CHANNELS, blk_channels=LOSS_BLK_CHANNELS)

# LOAD DATA
# trainset, valset = load_full_data_list(datafolder = datafolder)
# trainset, valset = load_full_data(trainset, valset)

SR = 16000
IN_MEMORY = 0.0
DIRECTORY = "/trainman-mount/trainman-storage-dc5e03f8-a08d-49bb-b3a9-4bae92eb4e92"
REVERB_DIRECTORY = "/trainman-mount/trainman-storage-420a420f-b7a2-4445-abca-0081fc7108ca/Audio-RIRs"
NOISE_DIRECTORY = "/trainman-mount/trainman-storage-420a420f-b7a2-4445-abca-0081fc7108ca/subnoises"
VAL_NOISE_DIRECTORY = "/trainman-mount/trainman-storage-420a420f-b7a2-4445-abca-0081fc7108ca/subnoises"
data_range = CLEAN_DATA_RANGE
test_data_range = CLEAN_TEST_DATA_RANGE

NO_CLASSES = 200
SEQ_LEN = 256000
BATCH_SIZE = 1
# LEARNING_RATE = config['LEARNING_RATE']
# path = config['path']
# dropout = config['dropout'] if "dropout" in config else 0.0
inject_noise = True  
use_real_noise = True
augment_speech = False
augment_reverb = False
# augment_noise = False
extra_subsets = False
    
#%% Speech audio files
train_filenames, train_data_holder = query_joint_yield(
                                         gender=data_range["gender"], 
                                         num=data_range["num"], 
                                         script=data_range["script"],
                                         device=data_range["device"], 
                                         scene=data_range["scene"], 
                                         directory=DIRECTORY, 
                                         exam_ignored=True, 
                                         randomized=True,
                                         sample_rate=SR, 
                                         in_memory=IN_MEMORY)

test_filenames, test_data_holder = query_joint_yield(
                                         gender=test_data_range["gender"], 
                                         num=test_data_range["num"], 
                                         script=test_data_range["script"],
                                         device=test_data_range["device"], 
                                         scene=test_data_range["scene"], 
                                         directory=DIRECTORY, 
                                         exam_ignored=True, 
                                         randomized=True,
                                         sample_rate=SR,
                                         in_memory=IN_MEMORY)
print(test_filenames)

if extra_subsets:
    unseen_speaker_test_filenames, _ = query_joint_yield(
                                         gender=test_data_range["gender"], 
                                         num=test_data_range["num"], 
                                         script=data_range["script"],
                                         device=test_data_range["device"], 
                                         scene=test_data_range["scene"], 
                                         directory=DIRECTORY, 
                                         exam_ignored=True, 
                                         randomized=True,
                                         sample_rate=SR,
                                         in_memory=IN_MEMORY)

    unseen_script_test_filenames, _ = query_joint_yield(
                                         gender=data_range["gender"], 
                                         num=data_range["num"], 
                                         script=test_data_range["script"],
                                         device=test_data_range["device"], 
                                         scene=test_data_range["scene"], 
                                         directory=DIRECTORY, 
                                         exam_ignored=True, 
                                         randomized=True,
                                         sample_rate=SR,
                                         in_memory=IN_MEMORY)

#%% Reverb audio files
# 1-250 Train
# 251-271 Test
# 233 is missing
reverb_train_filenames, reverb_train_data_holder=read_Audio_RIRs(sr=SR, 
                                                                 subset="train", 
                                                                 cutoff=NO_CLASSES, 
                                                                 root=REVERB_DIRECTORY)
reverb_test_filenames, reverb_test_data_holder=read_Audio_RIRs(sr=SR, 
                                                               subset="test", 
                                                               cutoff=NO_CLASSES, 
                                                               root=REVERB_DIRECTORY)

print(len(reverb_train_filenames), len(reverb_test_filenames))

#%% Target classes
class_dict, classes, class_back_dict = get_Audio_RIR_classes(REVERB_DIRECTORY, 271)

### Prepare Noise
if use_real_noise:
    noise_filenames, _ = read_noise(sr=SR, root=NOISE_DIRECTORY, preload=False)
#         print(noise_filenames)
    val_noise_filenames, _ = read_noise(sr=SR, root=VAL_NOISE_DIRECTORY, preload=False)
#     print(val_noise_filenames)
else:
    noise_filenames = None
    val_noise_filenames = None
    
train_set_generator = DataGenerator(train_filenames, reverb_train_filenames, noise_filenames=noise_filenames,
                                speech_data_holder=None, 
                                reverb_data_holder=None,
                                noise_data_holder=None,
                                sample_rate=SR, 
                                seq_len=SEQ_LEN, 
                                num_classes=NO_CLASSES,
                                shuffle=True, batch_size=BATCH_SIZE,
                                in_memory=1.0, 
                                augment_speech=augment_speech, inject_noise=inject_noise, augment_reverb=augment_reverb)

val_set_generator = DataGenerator(test_filenames, reverb_train_filenames, noise_filenames=val_noise_filenames,
                                speech_data_holder=None, 
                                reverb_data_holder=None,
                                noise_data_holder=None,
                                sample_rate=SR, 
                                seq_len=SEQ_LEN, 
                                num_classes=NO_CLASSES,
                                shuffle=True, batch_size=BATCH_SIZE,
                                in_memory=1.0, 
                                augment_speech=augment_speech, inject_noise=inject_noise, augment_reverb=augment_reverb)

if extra_subsets:
    val_unseen_speaker_set_generator = DataGenerator(unseen_speaker_test_filenames, reverb_train_filenames, 
                                    noise_filenames=val_noise_filenames,
                                    speech_data_holder=None, 
                                    reverb_data_holder=None,
                                    noise_data_holder=None,
                                    sample_rate=SR, 
                                    seq_len=SEQ_LEN, 
                                    num_classes=NO_CLASSES,
                                    shuffle=True, batch_size=BATCH_SIZE,
                                    in_memory=1.0, 
                                    augment_speech=augment_speech, inject_noise=inject_noise, augment_reverb=augment_reverb)

    val_unseen_script_set_generator = DataGenerator(unseen_script_test_filenames, reverb_train_filenames, 
                                    noise_filenames=val_noise_filenames,
                                    speech_data_holder=None, 
                                    reverb_data_holder=None,
                                    noise_data_holder=None,
                                    sample_rate=SR, 
                                    seq_len=SEQ_LEN, 
                                    num_classes=NO_CLASSES,
                                    shuffle=True, batch_size=BATCH_SIZE,
                                    in_memory=1.0, 
                                    augment_speech=augment_speech, inject_noise=inject_noise, augment_reverb=augment_reverb)

test_set_generator = DataGenerator(test_filenames, reverb_test_filenames, noise_filenames=val_noise_filenames,
                                    speech_data_holder=None, 
                                    reverb_data_holder=None,
                                    noise_data_holder=None,
                                    sample_rate=SR, 
                                    seq_len=SEQ_LEN, 
                                    num_classes=NO_CLASSES,
                                    shuffle=True, batch_size=BATCH_SIZE,
                                    in_memory=1.0, 
                                    augment_speech=augment_speech, inject_noise=inject_noise, augment_reverb=augment_reverb)

train_iter = train_set_generator.__iter__()
val_iter = val_set_generator.__iter__()
test_iter = test_set_generator.__iter__()

# TRAINING OPTIMIZER
opt=tf.train.AdamOptimizer(learning_rate=1e-4).\
    minimize(loss_fn[0],var_list=[var for var in tf.trainable_variables() if var.name.startswith("se_")])

# BEGIN SCRIPT #########################################################################################################

# INITIALIZE GPU CONFIG
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

print("Config ready")

sess.run(tf.global_variables_initializer())

print("Session initialized")

# LOAD FEATURE LOSS
if SE_LOSS_TYPE == "FL":
    loss_saver = tf.train.Saver([var for var in tf.trainable_variables() if var.name.startswith("loss_")])
    loss_saver.restore(sess, "./%s/loss_model.ckpt" % modfolder)

Nepochs = 500
saver = tf.train.Saver([var for var in tf.trainable_variables() if var.name.startswith("se_")])

if restore_from is not None:
    saver.restore(sess, restore_from)

########################################################################################################################

if SE_LOSS_TYPE == "FL":
    loss_train = np.zeros((train_set_generator.__len__(),SE_LOSS_LAYERS+1))
    loss_val = np.zeros((val_set_generator.__len__(),SE_LOSS_LAYERS+1))
    loss_test = np.zeros((test_set_generator.__len__(),SE_LOSS_LAYERS+1))
else:
    loss_train = np.zeros((train_set_generator.__len__(),1))
    loss_val = np.zeros((val_set_generator.__len__(),1))
    loss_test = np.zeros((test_set_generator.__len__(),1))
    
if SE_LOSS_TYPE == "FL":
    loss_w = np.ones(SE_LOSS_LAYERS)
else:
    loss_w = []

#####################################################################################

for epoch in range(1,Nepochs+1):

    print("Epoch no.%d"%epoch)
    # TRAINING EPOCH ################################################################

#     ids = np.random.permutation(len(trainset["innames"])) # RANDOM FILE ORDER

    for id in tqdm(range(0, train_set_generator.__len__()), file=sys.stdout):
        inputData, outputData = next(train_iter)
#         i = ids[id] # RANDOMIZED ITERATION INDEX
#         inputData = trainset["inaudio"][i] # LOAD DEGRADED INPUT
#         outputData = trainset["outaudio"][i] # LOAD GROUND TRUTH
            
        # TRAINING ITERATION
        _, loss_vec = sess.run([opt, loss_fn],
                                feed_dict={input: inputData, clean: outputData, loss_weights: loss_w})

        # SAVE ITERATION LOSS
        loss_train[id,0] = loss_vec[0]
        if SE_LOSS_TYPE == "FL":
            for j in range(SE_LOSS_LAYERS):
                loss_train[id,j+1] = loss_vec[j+1]

    # PRINT EPOCH TRAINING LOSS AVERAGE
    str = "T: %d\t " % (epoch)
    if SE_LOSS_TYPE == "FL":
        for j in range(SE_LOSS_LAYERS+1):
            str += ", %10.6e"%(np.mean(loss_train, axis=0)[j])
    else:
        str += ", %10.6e"%(np.mean(loss_train, axis=0)[0])

    log_file.write(str + "\n")
    log_file.flush()

    # SET WEIGHTS AFTER M EPOCHS
    if SE_LOSS_TYPE == "FL" and epoch == SET_WEIGHT_EPOCH:
        loss_w = np.mean(loss_train, axis=0)[1:]

    # SAVE MODEL EVERY N EPOCHS
    if epoch % SAVE_EPOCHS != 0:
        continue

    saver.save(sess, outfolder + "/se_model.ckpt")

    # VALIDATION EPOCH ##############################################################

    print("Validation epoch")
    for id in tqdm(range(0, val_set_generator.__len__()), file=sys.stdout):
        inputData, outputData = next(val_iter)
        
        # VALIDATION ITERATION
        output, loss_vec = sess.run([enhanced, loss_fn],
                            feed_dict={input: inputData, clean: outputData, loss_weights: loss_w})

        # SAVE ITERATION LOSS
        loss_val[id,0] = loss_vec[0]
        if SE_LOSS_TYPE == "FL":
            for j in range(SE_LOSS_LAYERS):
                loss_val[id,j+1] = loss_vec[j+1]

    # PRINT VALIDATION EPOCH LOSS AVERAGE
    str = "V: %d " % (epoch)
    if SE_LOSS_TYPE == "FL":
        for j in range(SE_LOSS_LAYERS+1):
            str += ", %10.6e"%(np.mean(loss_val, axis=0)[j]*1e9)
    else:
        str += ", %10.6e"%(np.mean(loss_val, axis=0)[0]*1e9)

    log_file.write(str + "\n")
    log_file.flush()
    
    # TEST EPOCH ##############################################################

    print("Test epoch")
    for id in tqdm(range(0, test_set_generator.__len__()), file=sys.stdout):
        inputData, outputData = next(test_iter)
        
        # VALIDATION ITERATION
        output, loss_vec = sess.run([enhanced, loss_fn],
                            feed_dict={input: inputData, clean: outputData, loss_weights: loss_w})

        # SAVE ITERATION LOSS
        loss_test[id,0] = loss_vec[0]
        if SE_LOSS_TYPE == "FL":
            for j in range(SE_LOSS_LAYERS):
                loss_test[id,j+1] = loss_vec[j+1]

    # PRINT VALIDATION EPOCH LOSS AVERAGE
    str = "E: %d " % (epoch)
    if SE_LOSS_TYPE == "FL":
        for j in range(SE_LOSS_LAYERS+1):
            str += ", %10.6e"%(np.mean(loss_test, axis=0)[j]*1e9)
    else:
        str += ", %10.6e"%(np.mean(loss_test, axis=0)[0]*1e9)

    log_file.write(str + "\n")
    log_file.flush()
    np.save("loss_w.npy", loss_w)

log_file.close()
