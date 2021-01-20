from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from matplotlib import cm, pyplot as plt
from sklearn import metrics
from os.path import expanduser as eu
from os.path import isfile, join
from os import listdir
import numpy as np
import random
import time
import keras
import os

# Make sure that the models only take GPU memory as needed
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=True,gpu_options = tf.GPUOptions(allow_growth=True))
set_session(tf.Session(config=config))

# Debug flag
DEBUG = True

# IDs associated with each feature
feat_ids = [2, 3, 10, 9, 1, 29, 27, 28, 7, 5, 8, 4, 0, 11, 6]

# List of feature names
X_lst = ['ECGRATE', 'ETCO2', 'ETSEV', 'ETSEVO', 'FIO2', 'NIBPD', 'NIBPM',
         'NIBPS', 'PEAK', 'PEEP', 'PIP', 'RESPRATE', 'SAO2', 'TEMP1', 'TV']

def normalize(data, reference):
    """
    Normalize data according to a reference data's mean 
    and standard deviation
    """
    return((data-reference.mean())/reference.std())
    
def impute(data, impute_per_sample=True):
    """
    Impute data
    
    Args:
     - data : data to impute
     - impute_per_samples : if true impute for each sample
    """
    # Impute for each sample
    if impute_per_sample:
        for i in range(data.shape[0]):
            nz_yi = data[i] != 0
            z_yi  = data[i] == 0
            data[i,z_yi] = data[i,nz_yi].mean()
    # Impute using the global median
    else:
        data[data==0] = np.median(data[data!=0])
    
def load_train_val_data(DATAPATH, 
                        task, 
                        feat=None, 
                        filter_zeros=True, 
                        skip_normalize=False, 
                        impute_per_sample=True):
    """
    Load train and validation data for a particular signal and 
    upstream embedding task
    
    Args:
     - DATAPATH : path data lives at
     - task : the upstream embedding task associated with the data
     - feat : the signal variable the data is associated with
     - filter_zeros : whether or not to filter out samples with all zero labels
     - skip_normalize : whether or not to normalize labels
    """
    
    print("[PROGRESS] Starting load_train_val_data()")
    
    if DEBUG: print("[DEBUG] Loading from DATAPATH: {}".format(DATAPATH))
    dfiles = os.listdir(DATAPATH)
    
    # Load x files
    def is_train_x(f,feat): return("X_train_60" in f and feat+".npy" in f)
    def is_val_x(f,feat):   return("X_val_60" in f and feat+".npy" in f)
    
    train_x_file = [f for f in dfiles if is_train_x(f,feat)][0]
    val_x_file   = [f for f in dfiles if is_val_x(f,feat)][0]

    if DEBUG:
        print("[DEBUG] Loading train_x_file: {}".format(train_x_file))
        print("[DEBUG] Loading val_x_file  : {}".format(val_x_file))
        
    train_x = np.load(DATAPATH+train_x_file, mmap_mode='r')
    val_x   = np.load(DATAPATH+val_x_file, mmap_mode='r')
    
    # Load y files
    if   task is "nextfive":
        def is_train_y(f,feat):  return("y_train_60" in f and feat+"nextfive" in f)
        def is_val_y(f,feat):    return("y_val_60" in f and feat+"nextfive" in f)
    elif task is "maximum5":
        def is_train_y(f,feat):  return("y_train_60" in f and feat+"nextfive" in f)
        def is_val_y(f,feat):    return("y_val_60" in f and feat+"nextfive" in f)
    elif task is "minimum5":
        def is_train_y(f,feat):  return("y_train_60" in f and "minimum5" in f)
        def is_val_y(f,feat):    return("y_val_60" in f and "minimum5" in f)
    elif task.startswith("hypo"):
        def is_train_y(f,feat):  return("y_train_60" in f)
        def is_val_y(f,feat):    return("y_val_60" in f)
        
    train_y_file = [f for f in dfiles if is_train_y(f,feat)][0]
    val_y_file   = [f for f in dfiles if is_val_y(f,feat)][0]

    # Check if a pre-processed version of the y labels exist
    train_y = np.load(DATAPATH+train_y_file, mmap_mode='r+')
    val_y   = np.load(DATAPATH+val_y_file, mmap_mode='r+')

    # Subset files
    if not subset_files is None:
        train_x, val_x, train_y, val_y = subset_data(DATAPATH, train_x, val_x, train_y, val_y, subset_files)
    
    # Add another axis
    train_x = train_x[:,:,np.newaxis]
    val_x   = val_x[:,:,np.newaxis]

    # Additional task-specific processing steps
    if task is "nextfive":
        # Represent nans as zeros
        if np.isnan(train_x).sum() > 0: train_x[np.isnan(train_x)] = 0
        if np.isnan(train_y).sum() > 0: train_y[np.isnan(train_y)] = 0
        if np.isnan(val_x).sum() > 0: val_x[np.isnan(val_x)] = 0
        if np.isnan(val_y).sum() > 0: val_y[np.isnan(val_y)] = 0

        # Filter out samples with all zero labels
        if filter_zeros:
            non_all_zero_ytrain = np.sum(np.abs(train_y),1) != 0
            non_all_zero_yval   = np.sum(np.abs(val_y),1) != 0
            train_x = train_x[non_all_zero_ytrain,]
            val_x   = val_x[non_all_zero_yval,]
            train_y = train_y[non_all_zero_ytrain,]
            val_y   = val_y[non_all_zero_yval,]

        # Impute the y values
        impute(train_y,impute_per_sample)
        impute(val_y,impute_per_sample)

    elif task in ["minimum5"] or task.startswith("hypo"):
        train_y = train_y[:,np.newaxis]
        val_y   = val_y[:,np.newaxis]

    if task == "maximum5":
        # Represent nans as zeros
        if np.isnan(train_x).sum() > 0: train_x[np.isnan(train_x)] = 0
        if np.isnan(train_y).sum() > 0: train_y[np.isnan(train_y)] = 0
        if np.isnan(val_x).sum() > 0: val_x[np.isnan(val_x)] = 0
        if np.isnan(val_y).sum() > 0: val_y[np.isnan(val_y)] = 0

        # Filter out samples with all zero labels
        if filter_zeros:
            non_all_zero_ytrain = np.sum(np.abs(train_y),1) != 0
            non_all_zero_yval   = np.sum(np.abs(val_y),1) != 0
            train_x = train_x[non_all_zero_ytrain,]
            val_x   = val_x[non_all_zero_yval,]
            train_y = train_y[non_all_zero_ytrain,]
            val_y   = val_y[non_all_zero_yval,]

        # Impute the y values
        impute(train_y,impute_per_sample)
        impute(val_y,impute_per_sample)

        # Use maximum of future five minutes as the labels
        train_y = train_y.max(1)
        val_y   = val_y.max(1)

        # Add another axis
        train_y = train_y[:,np.newaxis]
        val_y   = val_y[:,np.newaxis]

    if task in ["nextfive", "minimum5", "maximum5"]:
        if not skip_normalize:
            # Normalize - make sure to normalize the validation before the training
            val_y   = normalize(val_y,train_y)
            train_y = normalize(train_y,train_y)
        
    if DEBUG:
        print("[DEBUG] train_x.shape: {}, train_y.shape: {}".format(train_x.shape,train_y.shape))
        print("[DEBUG] val_x.shape  : {},   val_y.shape: {}".format(val_x.shape,val_y.shape))

    return(train_x, train_y, val_x, val_y)
    
def create_model(output_size=False, task="None", min_mod=None, lr=0.001, 
                 epoch_num=200, fine_tune=False):
    """
    Create an LSTM model with specific parameters
    """
    
    print("[PROGRESS] Starting create_model()")
    lookback = 60; h1 = 200; h2 = 200; b_size = 1000;
    if task.startswith("hypo"):
        opt_name = "rmsprop"
        opt = keras.optimizers.RMSprop(lr)
        loss_func = "binary_crossentropy"
    else:
        opt_name = "adam"
        opt = keras.optimizers.Adam(lr)
        loss_func = 'mean_squared_error'
    mod_name = "{}n_{}n_{}ep_{}ba_{}opt_{}loss".format(h1,h2,epoch_num,b_size,opt_name,loss_func)    
    
    model = Sequential()
    if min_mod is "randemb":
        model.add(LSTM(h1, recurrent_dropout=0.5, return_sequences=True, input_shape=(lookback,1)))
        model.add(LSTM(h2, recurrent_dropout=0.5,dropout=0.5))
        model.compile(loss=loss_func, optimizer=opt)
    elif not min_mod is None:
        model.add(LSTM(h1, recurrent_dropout=0.5, return_sequences=True, input_shape=(lookback,1), 
                        weights=min_mod.layers[0].get_weights()))
        model.add(LSTM(h2, recurrent_dropout=0.5,dropout=0.5,weights=min_mod.layers[1].get_weights()))
        if fine_tune: # Add the final layer only if we are fine tuning
            if task.startswith("hypo"):
                model.add(Dense(1, activation='sigmoid'))
            else:
                model.add(Dense(output_size))
        model.compile(loss=loss_func, optimizer=opt)
    else:
        model.add(LSTM(h1, recurrent_dropout=0.5, return_sequences=True, input_shape=(lookback,1)))
        model.add(LSTM(h2, recurrent_dropout=0.5, dropout=0.5))
        if task.startswith("hypo"):
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(output_size))
        model.compile(loss=loss_func, optimizer=opt)

    return(model,mod_name,epoch_num)
    
def train_model(model, mod_name, train_x, train_y, val_x, val_y, MODDIR, 
                epoch_num, early_stopping_rounds=5, per_iter_size=300000,
                verbose = 0, batch_size = 1000):
    print("[PROGRESS] Starting train_model()")
    if DEBUG: print("[DEBUG] MODDIR :{}".format(MODDIR))

    with open(MODDIR+"loss.txt", "w") as f:
        f.write("%s\t%s\t%s\t%s\n" % ("i", "train_loss", "val_loss", "epoch_time"))
        
    # Train and Save
    diffs = []
    best_loss_so_far = float("inf")
    start_time = time.time()

    non_improving_count = 0
    for i in range(0,epoch_num):
        if per_iter_size > train_x.shape[0]: per_iter_size = train_x.shape[0]
        inds = np.random.choice(train_x.shape[0],per_iter_size,replace=False)
        curr_x = train_x[inds,]; curr_y = train_y[inds,]

        history = model.fit(curr_x, curr_y, epochs=1, batch_size=batch_size, 
                            validation_data=(val_x,val_y),verbose=verbose)

        # Save details about training
        train_loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]
        epoch_time = time.time() - start_time
        with open(MODDIR+"loss.txt", "a") as f:
            f.write("%d\t%f\t%f\t%f\n" % (i, train_loss, val_loss, epoch_time))

        # Save model each iteration
        model.save("{}val_loss:{}_epoch:{}_{}.h5".format(MODDIR,val_loss,i,mod_name))

        # Keep track of best loss
        if (best_loss_so_far > val_loss):
            best_loss_so_far = val_loss
            non_improving_count = 0
        else:
            non_improving_count = non_improving_count+1
        if (non_improving_count >= early_stopping_rounds): break

def load_min_model_helper(MPATH):
    print("[PROGRESS] Starting load_min_model_helper()")
    print("[DEBUG] MPATH {}".format(MPATH))
    mfiles = os.listdir(MPATH)
    full_mod_name = MPATH.split("/")[-1]
    mfiles = [f for f in mfiles if "val_loss" in f]
    loss_lst = [float(f.split("val_loss:")[1].split("_")[0]) for f in mfiles]
    min_ind = loss_lst.index(min(loss_lst))
    min_mod_name = "{}/{}".format(MPATH,mfiles[min_ind])
    if DEBUG: print("[DEBUG] min_mod_name {}".format(mfiles[min_ind]))
    return(load_model(min_mod_name))
            
def load_min_model(MPATH,feat):
    print("[PROGRESS] Starting load_min_model()")    
    full_mod_names = os.listdir(MPATH)
    full_mod_name = [m for m in full_mod_names if "feat:"+feat+"_" in m][0]
    print("[DEBUG] full_mod_name {}".format(full_mod_name))
    return(load_min_model_helper(MPATH+full_mod_name))

def load_trval_test_data(DPATH,feat,dtype="reformat",subset_files=None):
    print("[PROGRESS] Starting load_trval_test_data()")    
    fnames = [f for f in os.listdir(DPATH) if "featname:{}".format(feat) in f]

    X_trval_fname = [f for f in fnames if "X_train_validation_60" in f][0]
    X_test1_fname = [f for f in fnames if "X_test1_60" in f][0]
    print("[DEBUG] X_trval_fname {}".format(X_trval_fname))
    print("[DEBUG] X_test1_fname {}".format(X_test1_fname))
    X_trval = np.load(DPATH+X_trval_fname, mmap_mode="r")
    X_test1 = np.load(DPATH+X_test1_fname, mmap_mode="r")

    if not subset_files is None:
        print("[PROGRESS] Subsetting")
        trval_file_inds = np.load(DPATH+"train_validation_fileinds.npy")
        assert trval_file_inds.shape[0] == X_trval.shape[0]
        if DEBUG: 
            print("[DEBUG] Original trval size: {}".format(X_trval.shape))
            print("[DEBUG] Number of procedures in trval: {}".format(np.unique(trval_file_inds).shape))
        X_trval = X_trval[np.isin(trval_file_inds, subset_files)]
        if DEBUG: 
            print("[DEBUG] Subsetted trval size: {}".format(X_trval.shape))
            print("[DEBUG] Number of subsetted procedures in trval {}".format(np.unique(trval_file_inds[np.isin(trval_file_inds, subset_files)]).shape))

    
    if dtype == "reformat":
        # Add another axis
        X_trval = X_trval[:,:,np.newaxis]
        X_test1 = X_test1[:,:,np.newaxis]

    if DEBUG: print("[DEBUG] X_trval.shape: {}, X_test1.shape: {}".format(X_trval.shape,X_test1.shape))
    return(X_trval,X_test1)

def embed_and_save(SPATH,suffix,model,X_trval,X_test1,feat,task):
    print("[PROGRESS] Starting embed_and_save()")    
    if not os.path.exists(SPATH): os.makedirs(SPATH)

    # Create embeddings and names
    trval_pred = model.predict(X_trval, batch_size=1000)
    test1_pred = model.predict(X_test1, batch_size=1000)
    trval_pred_name = "task:{}_feat:{}_trval_{}".format(task,feat,suffix)
    test1_pred_name = "task:{}_feat:{}_test1_{}".format(task,feat,suffix)

    if DEBUG:
        print("[DEBUG] Saving {}\nSaving {}".format(trval_pred_name,test1_pred_name))
        print("[DEBUG] Saving to {}".format(SPATH))
        print("[DEBUG] trval_pred_name {}\ntest1_pred_name {}".format(trval_pred_name,test1_pred_name))
        print("[DEBUG] trval_pred.shape {}, test1_pred.shape {}".format(trval_pred.shape, test1_pred.shape))

    # Save embeddings
    np.save(SPATH+trval_pred_name,trval_pred)
    np.save(SPATH+test1_pred_name,test1_pred)