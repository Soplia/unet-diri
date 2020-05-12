import os
import shutil

def createFloders(epoch, batch_size):
    #######################################################
    #Creating a Folder for every data of the program
    #######################################################
    New_folder = './model'
    if os.path.exists(New_folder) and os.path.isdir(New_folder):
        shutil.rmtree(New_folder)
    try:
        os.mkdir(New_folder)
    except OSError:
        print("Creation of the main directory '%s' failed " % New_folder)
    else:
        print("Successfully created the main directory '%s' " % New_folder)

    #######################################################
    #Setting the folder of saving the predictions
    #######################################################
    read_pred = './model/pred'
    if os.path.exists(read_pred) and os.path.isdir(read_pred):
        shutil.rmtree(read_pred)
    try:
        os.mkdir(read_pred)
    except OSError:
        print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
    else:
        print("Successfully created the prediction directory '%s' of dice loss" % read_pred)

    #######################################################
    #checking if the model exists and if true then delete
    #######################################################
    read_model_path = './model/Unet_D_' + str(epoch) + '_' + str(batch_size)
    if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
        shutil.rmtree(read_model_path)
        print('Model folder there, so deleted for newer one')
    try:
        os.mkdir(read_model_path)
    except OSError:
        print("Creation of the model directory '%s' failed" % read_model_path)
    else:
        print("Successfully created the model directory '%s' " % read_model_path)
