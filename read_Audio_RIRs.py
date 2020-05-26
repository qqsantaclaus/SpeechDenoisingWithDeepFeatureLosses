import librosa
import os
import json

exclude_list = ['h007_Bathroom_Small_41txts.wav', 
                'h024_Bathroom_9txts.wav', 
                'h036_Bathroom_5txts.wav', 
                'h041_TrainStation_SouthStationBoston_4txts.wav', 
                'h043_Train_BostonTRedLine_4txts.wav', 
                'h049_MallFoodCourt_BurlingtonMall_3txts.wav', 
                'h081_Shower_2txts.wav', 
                'h133_SubwayStation_ParkStreetBoston_1txts.wav', 
                'h151_Train_BostonTOrangeLine_1txts.wav', 
                'h174_Bar_1txts.wav', 
                'h190_Train_BostonTGreenline_1txts.wav', 
                'h213_SubwayStation_CentralSquareCambridge_1txts.wav', 
                'h271_Outside_InTramStopRainShelter_2txts.wav']

'''Not sorted, 1-1 correspondance between reverb_filenames and reverb_data_holder'''
def read_Audio_RIRs(sr, class_dict, subset_range=range(200)):
    # 0-199, 271-595 Train
    # 200-270 Test
    # "h233"/232 is missing
    reverb_data_holder = []
    reverb_filenames = []
    for key in class_dict.keys():
        if (int(key) in subset_range) and class_dict[key] is not None and (class_dict[key]["name"]+".wav" not in exclude_list):
            reverb_filenames.append(class_dict[key])
            y, _= librosa.core.load(class_dict[key]["filepath"], sr=sr, mono=True)
            reverb_data_holder.append(y)
    return reverb_filenames, reverb_data_holder
    
# '''Not sorted, 1-1 correspondance between reverb_filenames and reverb_data_holder'''
# def read_Audio_RIRs(sr, subset="train", cutoff=250, root="/mnt/ilcompfbd1/user/jsu/Audio-RIRs"):
#     # 1-250 Train
#     # 251-271 Test
#     # 233 is missing
#     reverb_data_holder = []
#     reverb_filenames = []
#     for subdir, dirs, files in os.walk(root):
#         for file in files:
#             #print os.path.join(subdir, file)
#             filepath = subdir + os.sep + file
#             if filepath.endswith(".wav"):
#                 number = int(file[1:4])
# #                 print(file, number)
#                 if (subset=="train" and number<=cutoff) or (subset=="test" and number>cutoff):
#                     if file not in exclude_list:
#                         reverb_filenames.append(filepath)
#                         y, _= librosa.core.load(filepath, sr=sr, mono=True)
#                         reverb_data_holder.append(y)
    
#     return reverb_filenames, reverb_data_holder

def get_Audio_RIR_classes(filepath):
    with open('rir_classes.json', 'r') as fp:
        class_dict = json.load(fp)
    
    classes = []
    for i in range(len(class_dict.keys())):
        if class_dict[str(i)] is not None:
            classes.append(class_dict[str(i)]["filepath"])
        else:
            classes.append("N/A")
#     print(classes)s
    
    return class_dict, classes
    
# def get_Audio_RIR_classes(root, num_classes):
#     class_dict = {}
#     for subdir, dirs, files in os.walk(root):
#         for file in files:
#             #print os.path.join(subdir, file)
#             filepath = subdir + os.sep + file
#             if filepath.endswith(".wav"):
#                 number = int(file[1:4])-1
#                 class_dict[number] = file
                
#     classes = []
#     for i in range(num_classes):
#         if i in class_dict:
#             classes.append(class_dict[i])
#         else:
#             classes.append("N/A")
#     print(classes)
    
#     return class_dict, classes

def read_noise(sr, root="/mnt/ilcompfbd1/user/jsu/reverb_tools_for_Generate_SimData/NOISE", preload=False):
    noise_data_loader = {}
    noise_filenames = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".wav"):
                noise_filenames.append(filepath)
                if preload:
                    y, _= librosa.core.load(filepath, sr=sr, mono=True)
                    noise_data_loader[filepath]=y
    
    return noise_filenames, noise_data_loader

