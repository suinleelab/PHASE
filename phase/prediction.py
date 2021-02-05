##############
### Set up ###
##############
from sklearn.preprocessing import StandardScaler
from os.path import expanduser as eu
from matplotlib import pyplot as plt
from numpy import *
from time import time as t
import numpy as np
import sklearn.metrics as metrics
import xgboost as xgb
import random
import sys
import os
import gc

DEBUG = False

non_signal_names1 = ['EKG_ASYST', 'gender_M', 'EKG_ts_AFIB', 'EKG_ts_AVPACED', 'EKG_LBBB', 'TOF_1-4', 'TOF_4-4',
'TOF_ts_4-4', 'TOF_ts_2-4', 'asaCodeEmergency', 'TOF_ts_3-4', 'EKG_STACH', 'weightPounds', 'TOF_ts_1-4',
'EKG_AVPACED', 'EKG_ts_SBRADY', 'TOF_0-4', 'EKG_ts_ASYST', 'TOF_ts_0-4', 'EKG_ts_NSR', 'TOF_4f', 'gender_F',
'asaCode', 'age', 'EKG_PVC', 'heightInches', 'EKG_NSR', 'TOF_ts_4f', 'EKG_ts_APACED', 'EKG_AFIB', 'EKG_AFLUT',
'EKG_ts_AFLUT', 'EKG_SBRAD', 'EKG_ts_nsr', 'EKG_VPACED', 'EKG_ts_SBRAD', 'EKG_ts_STACH', 'EKG_ts_PVC', 'EKG_ts_BBB',
'EKG_nsr', 'EKG_ts_NODAL', 'EKG_APACED', 'EKG_ts_VPACED', 'TOF_3-4', 'TOF_2-4', 'EKG_ts_LBBB', 'EKG_BBB',
'EKG_SBRADY', 'EKG_NODAL']

non_signal_names = ['gender_M', 'asaCodeEmergency', 'weightPounds', 'asaCode', 'age', 'heightInches']

allfeat_lst = ["SAO2", "FIO2", "ECGRATE", "ETCO2", "RESPRATE", "PEEP", "TV", "PEAK",
         "PIP", "ETSEVO", "ETSEV", "TEMP1", "PULSE", "O2FLOW", "TIDALVOLUME",
         "PEAKPRESSURE", "RATE", "AIRFLOW", "FINO", "N2OFLOW", "ABPM1",
         "ABPD1", "ABPS1", "BIS", "ETISO", "TEMP2", "EMG", "NIBPM", "NIBPS",
         "NIBPD", "ETDES", "CVP", "PAS", "PAD", "ICP", "weightPounds",
         "heightInches", "age", "gender", "asaCode", "hospital"]

top15 = ['ECGRATE', 'ETCO2', 'ETSEV', 'ETSEVO', 'FIO2', 'NIBPD', 'NIBPM',
         'NIBPS', 'PEAK', 'PEEP', 'PIP', 'RESPRATE', 'SAO2', 'TEMP1', 'TV']

#########################
### Code to load data ###
#########################

def load_raw_data(DPATH,
                  data_type,
                  dt,
                  X_ema,
                  non_signal_inds,
                  curr_feat,
                  is_single_feat,
                  rpath="raw",
                  stack_feat=True,
                  feat_lst=None):
    """
    Load raw data
    
    Args
     - DPATH : data path
     - data_type: embedding data type
     - dt : train or test data
     - X_ema : ema data (has static variables)
     - non_signal_inds : ema indices that are not signal variables
     - curr_feat : current feature corresponding to label_type
     - is_single_feat : whether to load a single feature
     - rpath : load raw or processed data (XGB vs MLP)
     - stack_feat : whether to concatenate features
     - feat_lst : list of features to load
     
    Returns
     - X : independent variables
    """
    
    if DEBUG:
        print("[DEBUG] Starting load_raw_data")
        print("[DEBUG] DPATH {}".format(DPATH))

    files = os.listdir("{}{}/".format(DPATH,rpath))
    X_files = np.array([f for f in files if "X_{}_".format(dt) in f and "featnum" in f])
    X_files = X_files[np.array([int(f.split("featnum:")[1].split("_")[0]) for f in X_files]) < 35]
    inds = [int(f.split("featnum:")[1].split("_")[0]) for f in X_files]
    X_files = [x for _,x in sorted(zip(inds,X_files))]
    
    if feat_lst is None: feat_lst = top15
    
    if is_single_feat:
        f = [f for f in X_files if curr_feat in f][0]
        if DEBUG: print("[DEBUG] Loading raw file: {}".format("{}{}/{}".format(DPATH,rpath,f)))
        X_curr = np.load("{}{}/{}".format(DPATH,rpath,f), mmap_mode="r")
        X = X_curr
        assert(X.shape[1] == 60)
    else:
        # Compile the X data into a list
        X_lst = []
        for i in range(0,len(X_files)):
            f = X_files[i]
            feat = f.split("featname:")[1].split(".npy")[0]
            if feat in feat_lst: # Check to make sure it's in the top 15
                if DEBUG: print("[DEBUG] Loading: {}{}/{}".format(DPATH,rpath,f))
                X_curr = np.load("{}{}/{}".format(DPATH,rpath,f), mmap_mode="r")
                X_lst.append(X_curr)

        # add the nonsignal stuff
        if "nonsignal" in data_type:
            X_ema_curr = X_ema[:,non_signal_inds]
            X_lst.append(X_ema_curr)

        # Concatenate
        if stack_feat:
            X = np.hstack(X_lst)
            if "nonsignal" in data_type:
                assert(X.shape[1] == len(feat_lst)*60 + len(non_signal_inds))
            else:
                assert(X.shape[1] == len(feat_lst)*60)
        else:
            X = X_lst
    return(X)

def load_ema_data(DPATH,
                  data_type,
                  dt,
                  X_ema,
                  non_signal_inds,
                  feat_lst_inds,
                  curr_feat,
                  curr_feat_inds_ema,
                  is_single_feat):
    """
    Load exponential moving average and variance data
    
    Args
     - DPATH : data path
     - data_type: embedding data type
     - dt : train or test data
     - X_ema : ema data (has static variables)
     - non_signal_inds : ema indices that are not signal variables
     - feat_lst_inds : features indices to load
     - curr_feat : current feature corresponding to label_type
     - curr_feat_inds_ema : indices corresponding to current feature
     - is_single_feat : whether to load a single feature
     
    Returns
     - X : independent variables
    """
    if is_single_feat:
        X = X_ema[:,curr_feat_inds_ema]
        assert(X.shape[1] == len(curr_feat_inds_ema))
    else:
        # Get the appropriate indices for the ema stuff
        curr_inds = feat_lst_inds
        if "nonsignal" in data_type:
            curr_inds = list(set(non_signal_inds) | set(feat_lst_inds))
            
        X = X_ema[:,curr_inds]
        if "nonsignal" in data_type:
            assert(X.shape[1] == len(feat_lst_inds) + len(non_signal_inds))
        else:
            assert(X.shape[1] == len(feat_lst_inds))
    return(X)

def list_files(DPATH,data_type,is_train):
    """ Helper function to list files in a particular directory
    """
    if "randemb" in data_type:
        hosp_model = 0
    else:
        hosp_model = data_type.split("_")[1].split("[")[0]
    if "min5" in data_type:
        HIDPATH = eu("{}hidden200/minimum5/model_{}/".format(DPATH,hosp_model))
    elif "auto" in data_type:
        HIDPATH = eu("{}hidden200/autoencoder/model_{}/".format(DPATH,hosp_model))
    elif "nextfivemulti" in data_type:
        HIDPATH = eu("{}hidden200/nextfivemulti/model_{}/".format(DPATH,hosp_model))
    elif "nextfive" in data_type:
        HIDPATH = eu("{}hidden200/nextfive/model_{}/".format(DPATH,hosp_model))
    elif "hypo" in data_type:
        HIDPATH = eu("{}hidden200/{}/model_{}/".format(DPATH,data_type.split("_")[0],hosp_model))
    elif "randemb" in data_type:
        HIDPATH = eu("{}hidden200/randemb/model_{}/".format(DPATH,hosp_model))

    files = os.listdir(HIDPATH)
    if DEBUG: print("[DEBUG] HIDPATH {}".format(HIDPATH))
    if is_train:
        hfiles = [f for f in files if "trval" in f or "train_val" in f]
    else:
        hfiles = [f for f in files if "test" in f]
    return(HIDPATH,hfiles,hosp_model)

def load_hid_data(DPATH,
                  data_type,
                  dt,
                  X_ema,
                  non_signal_inds,
                  is_train,
                  curr_feat,
                  hosp_data,
                  is_single_feat,
                  shrink_path=None,
                  feat_lst=None):
    """
    Load embedded (hidden) data
    
    Args
     - DPATH : data path
     - data_type: embedding data type
     - X_ema : ema data (has static variables)
     - non_signal_inds : ema indices that are not signal variables
     - is_train : whether to load train or test data
     - curr_feat : current feature corresponding to label_type
     - hosp_data : downstream hospital data set
     - is_single_feat : whether to load a single feature
     - shrink_path : path for shrinking data sets
     - feat_lst : list of features to load
     
    Returns
     - X : independent variables
    """
    if not shrink_path is None: DPATH = shrink_path
    if feat_lst is None: feat_lst = top15
    HIDPATH,hfiles,hosp_model = list_files(DPATH,data_type,is_train)
    if is_single_feat:
        fname = [f for f in hfiles if curr_feat in f][0]
        if DEBUG: print("[DEBUG] loading: {}".format(fname))
        X = np.load(HIDPATH+fname,mmap_mode="r")
        assert(X.shape[1] == 200)
    else:
        hidden_lst = []
        if hosp_model == "P":
            pre,suf = data_type.split("P")
            HIDPATH_hd, hfiles_hd, hd = list_files(DPATH,pre+str(hosp_data)+suf,is_train)
            assert hd == str(hosp_data)
            currfeat_fname_P = [f for f in hfiles if curr_feat in f][0]
            if DEBUG: print("[DEBUG] loading: {}".format(HIDPATH+currfeat_fname_P))
            currfeat_hidden = np.load(HIDPATH+currfeat_fname_P,mmap_mode="r")
            hidden_lst.append(currfeat_hidden)
            # Add all non-currfeat signals
            for feat_ind in range(0,len(allfeat_lst)):
                feat = allfeat_lst[feat_ind]
                if feat in feat_lst: # Make sure that we only grab the top15 features
                    if feat == curr_feat: continue
                    curr_file = [f for f in hfiles_hd if feat in f]
                    if curr_file == []: continue
                    print("[DEBUG] loading: {}".format(HIDPATH_hd+curr_file[0]))
                    feat_ind_hidden = np.load(HIDPATH_hd+curr_file[0],mmap_mode="r")
                    hidden_lst.append(feat_ind_hidden)            
        else:
            for feat_ind in range(0,len(allfeat_lst)):
                feat = allfeat_lst[feat_ind]
                if feat in feat_lst: # Make sure that we only grab the top15 features
    
                    # For running experiment with 14 features
                    if "nosao2" in data_type and feat is "SAO2": continue         
                    
                    curr_file = [f for f in hfiles if feat in f]
                    if curr_file == []: continue
                    feat_ind_hidden = np.load(HIDPATH+curr_file[0],mmap_mode="r")
                    if DEBUG: print("[DEBUG] loading: {}".format(HIDPATH+curr_file[0]))
                    hidden_lst.append(feat_ind_hidden)

        # add the nonsignal stuff
        if "nonsignal" in data_type:
            X_ema_curr = X_ema[:,non_signal_inds]
            hidden_lst.append(X_ema_curr)
        
        # For DEBUG purposes
        if DEBUG:
            print("[DEBUG] Shapes {}".format([hid.shape for hid in hidden_lst]))
            X = np.zeros(10)
        else:
            X = np.hstack(hidden_lst)
            num_feat = len(feat_lst)
            if "nosao2" in data_type: num_feat = len(feat_lst) - 1
                
            if "nonsignal" in data_type:
                assert(X.shape[1] == num_feat*200 + len(non_signal_inds))
            else:
                assert(X.shape[1] == num_feat*200)
    return(X)

def load_data(PATH,
              data_type,
              label_type,
              is_train,
              hosp_data,
              curr_feat,
              shrink_path=None,
              feat_lst=None):
    """
    Load data for downstream prediction models
    
    Args
     - PATH : default path
     - data_type : embedding type
     - label_type : downstream prediction type
     - is_train : loading train or test data
     - hosp_data : downstream hospital
     - curr_feat : current feature corresponding to label_type
     - shrink_path : path for shrinking data sets
     - feat_lst : list of features to use
     
    Returns
     - X : independent variables
     - y : dependent variables
    """
    
    is_single_feat = not "top15" in data_type
    DPATH = PATH+"/data/{}/hospital_{}/".format(label_type,hosp_data)
    if is_train:
        dt = "train_validation"
    else:
        dt = "test1"

    EMAPATH = eu("~/RNN/LSTM_Feature/code/{}/both_hospitals/ema/".format(label_type))
    ema_files  = os.listdir(EMAPATH)
    ema_fname  = [f for f in ema_files if "features-substandard-{}".format(dt) in f][0]
    feat_names = np.genfromtxt(EMAPATH+ema_fname,dtype='str')
    
    non_signal_inds = [i for i in range(0,len(feat_names)) if (any([t in feat_names[i] for t in non_signal_names]))]
    if "oldnonsignal" in data_type:
        non_signal_inds = [i for i in range(0,len(feat_names)) if (any([t in feat_names[i] for t in non_signal_names1]))]
        
    if feat_lst is None: feat_lst = top15
    is_feat_lst        = lambda f : any([t in f for t in feat_lst]) and not "+++" in f and not "---" in f
    feat_lst_inds      = [i for i in range(0,len(feat_names)) if is_feat_lst(feat_names[i])]
    curr_feat_inds_ema = [i for i in range(0,len(feat_names)) if curr_feat in feat_names[i]]
    
    X_ema = np.load(DPATH+"proc/X_{}_60_{}_ema.npy".format(dt,label_type), mmap_mode="r")
    Y     = np.load(DPATH+"raw/y_{}_60_{}.npy".format(dt,label_type), mmap_mode="r")
    
    if DEBUG: print("[DEBUG] Y.shape: {}".format(Y.shape))
    if   "raw" in data_type:
        X = load_raw_data(DPATH,data_type,dt,X_ema,non_signal_inds,curr_feat,is_single_feat,
                          feat_lst=feat_lst)
    elif "proc" in data_type:
        X = load_raw_data(DPATH,data_type,dt,X_ema,non_signal_inds,curr_feat,is_single_feat,
                          rpath="proc",stack_feat=False,feat_lst=feat_lst)
    elif "ema" in data_type:
        X = load_ema_data(DPATH,data_type,dt,X_ema,non_signal_inds,feat_lst_inds,curr_feat,
                          curr_feat_inds_ema,is_single_feat)
    elif "min5" in data_type or "auto" in data_type or "nextfive" in data_type or data_type.startswith("hypo") or "randemb" in data_type:
        X = load_hid_data(DPATH,data_type,dt,X_ema,non_signal_inds,is_train,curr_feat,
                          hosp_data,is_single_feat,
                          shrink_path=shrink_path,feat_lst=feat_lst)
    elif data_type == "rand_input[top15]+nonsignal":
        if DEBUG:
            print("[DEBUG] Prospective shape: {}".format((Y.shape[0],len(non_signal_inds) + 200*15)))
            X = zeros(10)
        else:
            X = np.random.normal(size=(Y.shape[0],len(non_signal_inds) + 200*15))
    else:
        print("[Error] Unsupported data_type: {}".format(data_type))
        
    if label_type == "med_phenyl":
        Y2 = np.copy(Y)
        Y2[Y2 != 0.0] = 1.0
        Y = Y2
    return(X,Y)

################################
### Code for downstream XGBs ###
################################

def train_xgb_model(RESDIR,
                    trainvalX,
                    trainvalY,
                    data_type,
                    label_type,
                    hosp_data,
                    eta):
    """
    Train downstream model
    
    Args
     - RESDIR : Directory to save trained model
     - trainvalX : train validation input data
     - trainvalY : train validation label data
     - data_type : the type of embedding associated with the data
     - label_type : downstream prediction outcome
     - hosp_data : the hospital we will load the data from
     - eta : learning rate
    """
    # Split Dataset into Training and Validation
    train_ratio = 0.9
    nine_tenths_ind = int(train_ratio*trainvalX.shape[0])
    trainX = trainvalX[0:nine_tenths_ind,:]
    valX = trainvalX[nine_tenths_ind:trainvalX.shape[0],:]
    trainY = trainvalY[0:nine_tenths_ind]
    valY = trainvalY[nine_tenths_ind:trainvalX.shape[0]]
    del trainvalX
    gc.collect()
    # Randomize
    indices = np.arange(0,trainX.shape[0])
    random.shuffle(indices)
    trainX = trainX[indices,:]
    trainY = trainY[indices]
    indices = np.arange(0,valX.shape[0])
    random.shuffle(indices)
    valX = valX[indices,:]
    valY = valY[indices]
    # Convert to xgb format
    dtrain = xgb.DMatrix(trainX, label=trainY)
    dvalid = xgb.DMatrix(valX, label=valY)
    del trainX, valX
    gc.collect()

    # Train Model and save it
    param = {'max_depth':6, 'eta':eta, 'subsample':0.5, 'gamma':1.0, 'min_child_weight':10,
             'base_score':sum(trainY)/len(trainY), 'objective':'binary:logistic', 'eval_metric':["logloss"]}

    save_path = RESDIR+"hosp{}_data/{}/".format(hosp_data,data_type)
    if not os.path.exists(save_path): os.makedirs(save_path)
    evallist = [(dvalid,"eval")]
    old_stdout = sys.stdout
    sys.stdout = open(save_path+'history.txt', 'w')
    bst = xgb.train(param, dtrain, 2000, evallist, early_stopping_rounds=5)
    sys.stdout = old_stdout
    bst.save_model(save_path+'mod_eta{}.model'.format(eta))
    del dtrain, dvalid, bst
    gc.collect()

def load_xgb_model_and_test(RESDIR,
                            test1X,
                            test1Y,
                            data_type,
                            label_type,
                            hosp_data,
                            eta,
                            hosp_model=None,
                            return_pred=False):
    """
    Load downstream model and evaluate on test data
    
    Args
     - RESDIR : directory to save results
     - test1X : testing input data
     - test1Y : testing label data
     - data_type : embedding type
     - label_type : downstream prediction outcome
     - hosp_data : hospital we draw the data from
     - eta : learning rate
     - hosp_model : hospital embeddings were trained in
     - return_pred : whether or not to return predictions for additional analyses
    """
    save_path = RESDIR+"hosp{}_data/{}/".format(hosp_data,data_type)
    bst = xgb.Booster()
    mod_path = save_path
    if not hosp_model is None:
        assert "raw" in data_type, "Currently implemented for raw only"
        mod_path = RESDIR+"hosp{}_data/raw[top15]+nonsignal/".format(hosp_model)
    if DEBUG: print("[DEBUG] Loading model from {}".format(mod_path))
    bst.load_model(mod_path+'mod_eta{}.model'.format(eta))
    
    # Evaluate by bootstrap resampling
    dtest = xgb.DMatrix(test1X)
    ypred = bst.predict(dtest)
    
    # If we just want to get the predictions
    if return_pred: return(ypred)
    
    auc_pr  = metrics.average_precision_score(test1Y, ypred)
    auc_roc = metrics.roc_auc_score(test1Y, ypred)
    np.random.seed(231)
    auc_pr_lst  = []
    auc_roc_lst = []
    for i in range(0,100):
        inds = np.random.choice(test1X.shape[0], test1X.shape[0], replace=True)
        auc_pr  = metrics.average_precision_score(test1Y[inds], ypred[inds])
        auc_roc = metrics.roc_auc_score(test1Y[inds], ypred[inds])
        auc_pr_lst.append(auc_pr)
        auc_roc_lst.append(auc_roc)
    auc_pr_lst  = np.array(auc_pr_lst)
    auc_roc_lst = np.array(auc_roc_lst)
    if DEBUG: print("[DEBUG] auc_pr_lst.mean(): {}".format(auc_pr_lst.mean()))
    if DEBUG: print("[DEBUG] auc_roc_lst.mean(): {}".format(auc_roc_lst.mean()))

    SP = RESDIR+"hosp{}_data/".format(hosp_data)
    f = open('{}conf_int_pr_hospdata{}.txt'.format(SP,hosp_data),'a')
    f.write("{}, {}+-{}\n".format(data_type,auc_pr_lst.mean().round(4),2*np.std(auc_pr_lst).round(4)))
    f.close()

    f = open('{}conf_int_roc_hospdata{}.txt'.format(SP,hosp_data),'a')
    f.write("{}, {}+-{}\n".format(data_type,auc_roc_lst.mean().round(4),2*np.std(auc_roc_lst).round(4)))
    f.close()
    
    if not os.path.exists(save_path): os.makedirs(save_path)
    if DEBUG: print("[DEBUG] Saving results at {}".format(save_path))
    np.save("{}auc_pr_lst".format(save_path,data_type), auc_pr_lst)
    np.save("{}auc_roc_lst".format(save_path,data_type), auc_roc_lst)
    del test1X, dtest, bst
    gc.collect()
    
################################
### Code for downstream MLPs ###
################################
    
def load_min_model_helper(MPATH):
    """ Helper function to load best validation performance model
    """
    
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

def train_mlp_model(RESDIR,
                    trainvalX,
                    trainvalY,
                    data_type,
                    label_type,
                    hosp_data):
    """
    Train downstream MLP model
    
    Args
     - RESDIR : Directory to save trained model
     - trainvalX : train validation input data
     - trainvalY : train validation label data
     - data_type : the type of embedding associated with the data
     - label_type : downstream prediction outcome
     - hosp_data : the hospital we will load the data from
    """
    
    train_ratio = 0.9
    nine_tenths_ind = int(train_ratio*trainvalX.shape[0])
    X_train = trainvalX[0:nine_tenths_ind,:]
    y_train = trainvalY[0:nine_tenths_ind]
    X_valid = trainvalX[nine_tenths_ind:trainvalX.shape[0],:]
    y_valid = trainvalY[nine_tenths_ind:trainvalX.shape[0]]
    del trainvalX
    gc.collect()
    # Randomize
    indices = np.arange(0,X_train.shape[0])
    random.shuffle(indices)
    X_train = X_train[indices,:]
    y_train = y_train[indices]

    indices = np.arange(0,X_valid.shape[0])
    random.shuffle(indices)
    X_valid = X_valid[indices,:]
    y_valid = y_valid[indices]

    print("[PROGRESS] Starting create_model()")
    # lookback = 60; h1 = 200; h2 = 200;
    b_size = 1000; epoch_num = 200; lr = 0.00001
    opt_name = "adam"
    opt = keras.optimizers.Adam(lr)
    loss_func = "binary_crossentropy"
    mod_name = "multivariate_mlp_label{}_dtype{}_hd{}".format(label_type,data_type,hosp_data)
    mod_name += "_{}ep_{}ba_{}opt_{}loss".format(epoch_num,b_size,opt_name,loss_func)

    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss_func, optimizer=opt)

    MODDIR = PATH+"models/"+mod_name+"/"
    if not os.path.exists(MODDIR): os.makedirs(MODDIR)

    with open(MODDIR+"loss.txt", "w") as f:
        f.write("%s\t%s\t%s\t%s\n" % ("i", "train_loss", "val_loss", "epoch_time"))

    # Train and Save
    diffs = []; best_loss_so_far = float("inf")
    start_time = time.time(); per_iter_size = 300000
    for i in range(0,epoch_num):
        if per_iter_size < X_train.shape[0]:
            per_iter_size = X_train.shape[0]
        inds = np.random.choice(X_train.shape[0],per_iter_size,replace=False)
        curr_x = X_train[inds,]; curr_y = y_train[inds,]
        history = model.fit(curr_x, curr_y, epochs=1, batch_size=1000, 
                            validation_data=(X_valid,y_valid))

        # Save details about training
        train_loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]
        epoch_time = time.time() - start_time
        with open(MODDIR+"loss.txt", "a") as f:
            f.write("%d\t%f\t%f\t%f\n" % (i, train_loss, val_loss, epoch_time))

        # Save model each iteration
        model.save("{}val_loss:{}_epoch:{}_{}.h5".format(MODDIR,val_loss,i,mod_name))
    return(MODDIR)

def load_mlp_model_and_test(RESDIR,
                            MODDIR,
                            X_test,
                            y_test,
                            data_type,
                            label_type,
                            hosp_data):
    """
    Load downstream model and evaluate on test data
    
    Args
     - RESDIR : directory to save results
     - MODDIR : directory model lives in
     - X_test : testing input data
     - y_test : testing label data
     - data_type : embedding type
     - label_type : downstream prediction outcome
     - hosp_data : hospital we draw the data from
    """
    model = load_min_model_helper(MODDIR)
    save_path = RESDIR+"hosp{}_data/{}/".format(hosp_data,data_type)
    if not os.path.exists(save_path): os.makedirs(save_path)
    print("[DEBUG] Loading model from {}".format(save_path))
    ypred = model.predict(X_test)
    np.save(save_path+"ypred.npy",ypred)
    np.save(save_path+"y_test.npy",y_test)
    auc = metrics.average_precision_score(y_test, ypred)
    np.random.seed(231)
    auc_lst = []
    roc_auc_lst = []
    for i in range(0,100):
        inds = np.random.choice(X_test.shape[0], X_test.shape[0], replace=True)
        auc = metrics.average_precision_score(y_test[inds], ypred[inds])
        auc_lst.append(auc)
        roc_auc = metrics.roc_auc_score(y_test[inds], ypred[inds])
        roc_auc_lst.append(roc_auc)
    auc_lst = np.array(auc_lst)
    roc_auc_lst = np.array(roc_auc_lst)
    print("[DEBUG] auc_lst.mean(): {}".format(auc_lst.mean()))
    print("[DEBUG] roc_auc_lst.mean(): {}".format(roc_auc_lst.mean()))

    SP = RESDIR+"hosp{}_data/".format(hosp_data)
    f = open('{}conf_int_hospdata{}_prauc.txt'.format(SP,hosp_data),'a')
    f.write("{}, {}+-{}\n".format(data_type,auc_lst.mean().round(4),2*np.std(auc_lst).round(4)))
    f.close()
    f = open('{}conf_int_hospdata{}_rocauc.txt'.format(SP,hosp_data),'a')
    f.write("{}, {}+-{}\n".format(data_type,roc_auc_lst.mean().round(4),2*np.std(roc_auc_lst).round(4)))
    f.close()
    np.save("{}auc_lst".format(save_path,data_type), auc_lst)
    np.save("{}roc_auc_lst".format(save_path,data_type), roc_auc_lst)
    
#################################
### Code for downstream LSTMs ###
#################################
    
def standardize_static(X_lst):
    """ Standardize static features
    """
    mean = X_lst[-1].mean(0)
    std  = X_lst[-1].std(0)
    std[std == 0] = 1
    X_lst[-1] = (X_lst[-1] - mean)/std
    return(X_lst)

def split_data_lst(X_trval_lst,y_trval):
    """ Split data in list form
    """
    train_ratio = 0.9
    sample_size = X_trval_lst[0].shape[0]
    nine_tenths_ind = int(train_ratio*sample_size)
    X_train = [X[0:nine_tenths_ind,:] for X in X_trval_lst]
    y_train = y_trval[0:nine_tenths_ind]
    X_valid = [X[nine_tenths_ind:sample_size,:] for X in X_trval_lst]
    y_valid = y_trval[nine_tenths_ind:sample_size]
    del X_trval_lst
    gc.collect()
    return(X_train, y_train, X_valid, y_valid)

def form_LSTM_model_train(X_trval_lst,
                          y_trval,
                          hyper_dict,
                          label_type,
                          epoch_num=50,
                          is_tune=True):
    """
    Form LSTM model and train
    
    Args
     - X_trval_lst : train validation data in list form
     - y_trval : train validation output in list form
     - hyper_dict : dictionary of hyperparameters
     - label_type : downstream prediction outcome
     - epoch_num : number of epochs
     - is_tune : whether to fine tune
    """
    ########## Form Data #########
    X_train_lst, y_train, X_valid_lst, y_valid = split_data_lst(X_trval_lst,y_trval)

    X_train_lst = [X[:,:,np.newaxis] if X.shape[1] == 60 else X for X in X_train_lst]
    X_train_lst = [np.concatenate(X_train_lst[:-1],2),X_train_lst[-1]]

    X_valid_lst = [X[:,:,np.newaxis] if X.shape[1] == 60 else X for X in X_valid_lst]
    X_valid_lst = [np.concatenate(X_valid_lst[:-1],2),X_valid_lst[-1]]

    ########## Form Model #########
    print("[PROGRESS] form_model()")

    # Hyperparameters
    numlayer = hyper_dict["numlayer"]
    nodesize = hyper_dict["numnode"]
    opt_name = hyper_dict["opt"]
    drop     = hyper_dict["drop"]
    lr       = hyper_dict["lr"]
    
    if opt_name == "RMSprop":
        opt = RMSprop(lr);
    elif opt_name == "Adam":
        opt = Adam(lr);
    elif opt_name == "SGD":
        opt = SGD(lr);
    
    b_size = 1000
    per_epoch_size = 300000

    lookback = 60
    loss_func = "binary_crossentropy"

    # Model name
    if is_tune:
        mod_path  = "tune_multilstm_{}hospdata_{}label".format(hosp_data,label_type)
        mod_name  = "".join(["{}{}_".format(hyper_dict[k],k) for k in hyper_dict])
        mod_name += "{}bsize_{}epochnum_{}perepochsize".format(b_size,epoch_num,per_epoch_size)
        MODDIR = "{}models/{}/{}/".format(PATH,mod_path,mod_name)
    else:
        mod_name  = "multilstm_{}hospdata_{}label_".format(hosp_data,label_type)
        mod_name += "".join(["{}{}_".format(hyper_dict[k],k) for k in hyper_dict])
        mod_name += "{}bsize_{}epochnum_{}perepochsize".format(b_size,epoch_num,per_epoch_size)
        MODDIR = "{}models/{}/".format(PATH,mod_name)
    
    input_lst   = []; encoded_lst = []
    # Signals
    sig = Input(shape=(lookback,15))
    if numlayer == 1:
        lstm1 = LSTM(nodesize, recurrent_dropout=drop)
        encoded = lstm1(sig)
    elif numlayer == 2:
        lstm1 = LSTM(nodesize, recurrent_dropout=drop, return_sequences=True)
        lstm2 = LSTM(nodesize, recurrent_dropout=drop, dropout=drop)
        encoded = lstm2(lstm1(sig))
    elif numlayer == 3:
        lstm1 = LSTM(nodesize, recurrent_dropout=drop, return_sequences=True)
        lstm2 = LSTM(nodesize, recurrent_dropout=drop, dropout=drop, return_sequences=True)
        lstm3 = LSTM(nodesize, recurrent_dropout=drop, dropout=drop)
        encoded = lstm3(lstm2(lstm1(sig)))
    else:
        assert numlayer <= 3, "Too many layers"
    input_lst.append(sig); encoded_lst.append(encoded)

    # Static variables
    static_size = X_trval_lst[-1].shape[1]
    static = Input(shape=[static_size])
    input_lst   += [static]; encoded_lst += [static]
    
    # Combine and compile model
    merged_vector = keras.layers.concatenate(encoded_lst, axis=-1)
    predictions = Dense(1, activation='sigmoid')(merged_vector)
    model = Model(inputs=input_lst, outputs=predictions)
    model.compile(optimizer=opt, loss=loss_func)
    
    ########## Training #########
    print("[PROGRESS] Starting train_model()")

    loss_keys = ["loss", 'val_loss']

    print("[PROGRESS] Making MODDIR {}".format(MODDIR))
    if not os.path.exists(MODDIR): os.makedirs(MODDIR)
    with open(MODDIR+"loss.txt", "w") as f:
        f.write("\t".join(["i", "epoch_time"] + loss_keys)+"\n")

    # Train and Save
    start_time = time.time()
    for i in range(0,epoch_num):
        train_subset_inds = np.random.choice(X_train_lst[0].shape[0],per_epoch_size,replace=False)
#         pos_inds = np.where(y_train==1)[0]
#         neg_inds = np.where(y_train!=1)[0]
#         neg_subset_inds = np.random.choice(neg_inds,pos_inds.shape[0],replace=False)
#         train_subset_inds = np.concatenate([pos_inds,neg_subset_inds])
        np.random.shuffle(train_subset_inds)
        X_train_lst_sub = [X[train_subset_inds] for X in X_train_lst]
        y_train_sub     = y_train[train_subset_inds]
        history = model.fit(X_train_lst_sub, y_train_sub, epochs=1, batch_size=b_size, 
                            validation_data=(X_valid_lst,y_valid))

        # Save details about training
        epoch_time = time.time() - start_time
        write_lst = [i, epoch_time]
        write_lst += [history.history[k][0] for k in loss_keys]
        with open(MODDIR+"loss.txt", "a") as f:
            f.write("\t".join([str(round(e,5)) for e in write_lst])+"\n")

        # Save model each iteration
        val_loss = history.history['val_loss'][0]
        model.save("{}val_loss:{}_epoch:{}.h5".format(MODDIR,val_loss,i))

    return(MODDIR)

def load_LSTM_model_and_test(RESDIR,
                             MODDIR,
                             X_test,
                             y_test,
                             data_type,
                             hosp_data):
    """
    Load the LSTM model and evaluate on the test set
    
    Args
     - RESDIR : result directory
     - MODDIR : model directory
     - X_test : independent variables in test set
     - y_test : dependent variables in test set
     - data_type : embedding type
     - hosp_data : downstream hospital data set
    """
    model = load_min_model_helper(MODDIR)
    save_path = RESDIR+"hosp{}_data/{}/".format(hosp_data,data_type)
    if not os.path.exists(save_path): os.makedirs(save_path)
    print("[DEBUG] Loading model from {}".format(save_path))
    ypred = model.predict(X_test)
    np.save(save_path+"ypred.npy",ypred)
    np.save(save_path+"y_test.npy",y_test)
    auc = metrics.average_precision_score(y_test, ypred)
    np.random.seed(231)
    auc_lst = []
    roc_auc_lst = []
    for i in range(0,100):
        inds = np.random.choice(X_test[-1].shape[0], X_test[-1].shape[0], replace=True)
        auc = metrics.average_precision_score(y_test[inds], ypred[inds])
        auc_lst.append(auc)
        roc_auc = metrics.roc_auc_score(y_test[inds], ypred[inds])
        roc_auc_lst.append(roc_auc)
    auc_lst = np.array(auc_lst)
    roc_auc_lst = np.array(roc_auc_lst)
    print("[DEBUG] auc_lst.mean(): {}".format(auc_lst.mean()))
    print("[DEBUG] roc_auc_lst.mean(): {}".format(roc_auc_lst.mean()))

    SP = RESDIR+"hosp{}_data/".format(hosp_data)
    f = open('{}conf_int_hospdata{}_prauc.txt'.format(SP,hosp_data),'a')
    f.write("{}, {}+-{}\n".format(data_type,auc_lst.mean().round(4),2*np.std(auc_lst).round(4)))
    f.close()
    f = open('{}conf_int_hospdata{}_rocauc.txt'.format(SP,hosp_data),'a')
    f.write("{}, {}+-{}\n".format(data_type,roc_auc_lst.mean().round(4),2*np.std(roc_auc_lst).round(4)))
    f.close()
    np.save("{}auc_lst".format(save_path,data_type), auc_lst)
    np.save("{}roc_auc_lst".format(save_path,data_type), roc_auc_lst)