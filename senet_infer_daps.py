from model import *
from data_import import *

import sys, getopt
import soundfile as sf

modfolder = "models"
output_path = "sim_denoised"
try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:m:",["ifolder=,modelfolder="])
except getopt.GetoptError:
    print('Usage: python senet_infer.py -d <inputfolder> -m <modelfolder>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('Usage: pythonsenet_infer.py -d <inputfolder> -m <modelfolder>')
        sys.exit()
    elif opt in ("-m", "--modelfolder"):
        modfolder = arg
print("Model folder is " + modfolder + "/")


# SPEECH ENHANCEMENT NETWORK
SE_LAYERS = 13 # NUMBER OF INTERNAL LAYERS
SE_CHANNELS = 64 # NUMBER OF FEATURE CHANNELS PER LAYER
SE_LOSS_LAYERS = 6 # NUMBER OF FEATURE LOSS LAYERS
SE_NORM = "NM" # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

fs = 16000

# SET LOSS FUNCTIONS AND PLACEHOLDERS
with tf.variable_scope(tf.get_variable_scope()):
    input=tf.placeholder(tf.float32,shape=[None,1,None,1])
    clean=tf.placeholder(tf.float32,shape=[None,1,None,1])
        
    enhanced=senet(input, n_layers=SE_LAYERS, norm_type=SE_NORM, n_channels=SE_CHANNELS)

# BEGIN SCRIPT #########################################################################################################

# INITIALIZE GPU CONFIG
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

print("Config ready")

sess.run(tf.global_variables_initializer())

print("Session initialized")

saver = tf.train.Saver([var for var in tf.trainable_variables() if var.name.startswith("se_")])
saver.restore(sess, "./%s/se_model.ckpt" % modfolder)

#####################################################################################
# data_root_path = "/trainman-mount/trainman-storage-420a420f-b7a2-4445-abca-0081fc7108ca/daps_SE_results_cuts"
data_root_path = "/trainman-mount/trainman-storage-420a420f-b7a2-4445-abca-0081fc7108ca/sim_SE_results_cuts"

env_sel = np.load("/home/code-base/runtime/experiments/helper_notebooks/sim_selections.npy")

sp_envs_results = {}

for (dirpath, dirnames, filenames) in os.walk(data_root_path):
    if not (dirpath.split("/")[-1]=="Reverb"):
        continue
    sp_env = dirpath.split("/")[-2]
    ref_filename_list = []
    enhanced_filename_List = []
    for filename in filenames:
        if not filename.endswith(".wav"):
            continue
        if not filename in env_sel:
            continue
        enhanced_path = os.path.join(output_path, modfolder.split("/")[-1], sp_env, "DeepFL-Pre")
        enhanced_name = os.path.join(output_path, modfolder.split("/")[-1], sp_env, "DeepFL-Pre", filename.replace("Reverb", "DeepFL-Pre"))
        print(enhanced_name)
        if not os.path.exists(enhanced_path):
            os.makedirs(enhanced_path)
        
        noisy_wav, _ = librosa.load(os.path.join(dirpath, filename), sr=16000, mono=True)
        
        noisy_wav = np.reshape(noisy_wav, (1, 1, noisy_wav.shape[0], 1))
        output = sess.run([enhanced], feed_dict={input: noisy_wav})
        output = np.reshape(output, -1)
        sf.write(enhanced_name, output, 16000)