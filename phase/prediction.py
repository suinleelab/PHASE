########################
### Set up
########################
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

def get_subset_inds(DPATH, subset_fileinds):
    trval_files = np.load("{}raw/train_validation_fileinds.npy".format(DPATH))
    X_inds = np.isin(trval_files, subset_fileinds)
    return(X_inds)

def load_raw_data(DPATH,data_type,dt,X_ema,non_signal_inds,curr_feat,is_single_feat,
                  subset_fileinds=None,rpath="raw",stack_feat=True,feat_lst=None):
    
    if DEBUG:
        print("[DEBUG] Starting load_raw_data")
        print("[DEBUG] DPATH {}".format(DPATH))

    files = os.listdir("{}{}/".format(DPATH,rpath))
    X_files = np.array([f for f in files if "X_{}_".format(dt) in f and "featnum" in f])
    X_files = X_files[np.array([int(f.split("featnum:")[1].split("_")[0]) for f in X_files]) < 35]
    inds = [int(f.split("featnum:")[1].split("_")[0]) for f in X_files]
    X_files = [x for _,x in sorted(zip(inds,X_files))]
    
    if feat_lst is None: feat_lst = top15
    
    if not subset_fileinds is None: X_inds = get_subset_inds(DPATH, subset_fileinds)

    if is_single_feat:
        f = [f for f in X_files if curr_feat in f][0]
        if DEBUG: print("[DEBUG] Loading raw file: {}".format("{}{}/{}".format(DPATH,rpath,f)))
        X_curr = np.load("{}{}/{}".format(DPATH,rpath,f), mmap_mode="r")
        if not subset_fileinds is None: X_curr = X_curr[X_inds,:]
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
                if not subset_fileinds is None: X_curr = X_curr[X_inds,:]
                X_lst.append(X_curr)

        # add the nonsignal stuff
        if "nonsignal" in data_type:
            X_ema_curr = X_ema[:,non_signal_inds]
            if not subset_fileinds is None:
                X_ema_curr = X_ema_curr[X_inds]
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

def load_ema_data(DPATH,data_type,dt,X_ema,non_signal_inds,feat_lst_inds,curr_feat,curr_feat_inds_ema,
                  is_single_feat,subset_fileinds):
    if is_single_feat:
        X = X_ema[:,curr_feat_inds_ema]
        if not subset_fileinds is None: 
            X_inds = get_subset_inds(DPATH, subset_fileinds)
            X = X[X_inds]
        assert(X.shape[1] == len(curr_feat_inds_ema))
    else:
        # Get the appropriate indices for the ema stuff
        curr_inds = feat_lst_inds
        if "nonsignal" in data_type:
            curr_inds = list(set(non_signal_inds) | set(feat_lst_inds))
            
        X = X_ema[:,curr_inds]
        if not subset_fileinds is None: 
            X_inds = get_subset_inds(DPATH, subset_fileinds)
            X = X[X_inds]
        if "nonsignal" in data_type:
            assert(X.shape[1] == len(feat_lst_inds) + len(non_signal_inds))
        else:
            assert(X.shape[1] == len(feat_lst_inds))
    return(X)

def list_files(DPATH,data_type,is_train):
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

def load_hid_data(DPATH,data_type,dt,X_ema,non_signal_inds,is_train,
                  curr_feat,hosp_data,is_single_feat,subset_fileinds=None,
                  shrink_path=None,feat_lst=None):
    if not subset_fileinds is None: X_inds = get_subset_inds(DPATH, subset_fileinds)
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
                    if not subset_fileinds is None and shrink_path is None:
                        feat_ind_hidden = feat_ind_hidden[X_inds,:]
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
                    if not subset_fileinds is None and shrink_path is None: 
                        feat_ind_hidden = feat_ind_hidden[X_inds,:]
                    hidden_lst.append(feat_ind_hidden)

        # add the nonsignal stuff
        if "nonsignal" in data_type:
            X_ema_curr = X_ema[:,non_signal_inds]
            if not subset_fileinds is None:
                X_ema_curr = X_ema_curr[X_inds]
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

def load_data(PATH,data_type,label_type,is_train,hosp_data,curr_feat,
              subset_fileinds=None,shrink_path=None,feat_lst=None):
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
    if not subset_fileinds is None: 
        X_inds = get_subset_inds(DPATH, subset_fileinds)
        Y = Y[X_inds]
    
    if DEBUG: print("[DEBUG] Y.shape: {}".format(Y.shape))
    if   "raw" in data_type:
        X = load_raw_data(DPATH,data_type,dt,X_ema,non_signal_inds,curr_feat,is_single_feat,
                          subset_fileinds=subset_fileinds,feat_lst=feat_lst)
    elif "proc" in data_type:
        X = load_raw_data(DPATH,data_type,dt,X_ema,non_signal_inds,curr_feat,is_single_feat,
                          subset_fileinds=subset_fileinds,rpath="proc",stack_feat=False,feat_lst=feat_lst)
    elif "ema" in data_type:
        X = load_ema_data(DPATH,data_type,dt,X_ema,non_signal_inds,feat_lst_inds,curr_feat,
                          curr_feat_inds_ema,is_single_feat,subset_fileinds=subset_fileinds)
    elif "min5" in data_type or "auto" in data_type or "nextfive" in data_type or data_type.startswith("hypo") or "randemb" in data_type:
        X = load_hid_data(DPATH,data_type,dt,X_ema,non_signal_inds,is_train,curr_feat,
                          hosp_data,is_single_feat,subset_fileinds=subset_fileinds,
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

def train_xgb_model(RESDIR,trainvalX,trainvalY,data_type,label_type,hosp_data,eta):
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

def load_xgb_model_and_test(RESDIR,test1X,test1Y,data_type,label_type,
                            hosp_data,xgb_type,eta,hosp_model=None,
                            return_pred=False):
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