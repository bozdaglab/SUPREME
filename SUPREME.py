# Options
addRawFeat = True
base_path = ''
feature_networks_integration = ['clinical', 'cna', 'exp']
node_networks = ['clinical', 'cna', 'exp']
int_method = 'MLP' # 'MLP' or 'XGBoost' or 'RF' or 'SVM'
xtimes = 50 
xtimes2 = 10 

feature_selection_per_network = [False, False, False]
top_features_per_network = [50, 50, 50]
optional_feat_selection = False
boruta_runs = 100
boruta_top_features = 50

max_epochs = 500
min_epochs = 200
patience = 30
learning_rates = [0.01, 0.001, 0.0001]
hid_sizes = [16, 32, 64, 128, 256, 512] 

random_state = 404

# SUPREME run
print('SUPREME is setting up!')
from lib import module, function
import time
import os, pickle, pyreadr, itertools
from sklearn.metrics import f1_score, accuracy_score
import statistics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, RandomizedSearchCV, GridSearchCV
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import os
import torch
import argparse
import errno
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if ((True in feature_selection_per_network) or (optional_feat_selection == True)):
    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    utils = importr('utils')
    rFerns = importr('rFerns')
    Boruta = importr('Boruta')
    pracma = importr('pracma')
    dplyr = importr('dplyr')
    import re

# Parser
parser = argparse.ArgumentParser(description='''An integrative node classification framework, called SUPREME 
(a subtype prediction methodology), that utilizes graph convolutions on multiple datatype-specific networks that are annotated with multiomics datasets as node features. 
This framework is model-agnostic and could be applied to any classification problem with properly processed datatypes and networks.
In our work, SUPREME was applied specifically to the breast cancer subtype prediction problem by applying convolution on patient similarity networks
constructed based on multiple biological datasets from breast tumor samples.''')
parser.add_argument('-csv', "--convert_csv", help="Converts csv files in the input directory to pkl files.", action="store_true")
parser.add_argument('-data', "--data_location", nargs = 1, default = ['sample_data'])
parser.add_argument('-gpu', "--gpu_support", help="Enables GPU support.", action="store_true", default = False)
parser.add_argument('-gpu_id', "--gpu_id_to_use", nargs = 1, default = [0])

args = parser.parse_args()
dataset_name = args.data_location[0]
enable_CUDA = args.gpu_support
gpu_id = args.gpu_id_to_use[0]

path = base_path + "data/" + dataset_name
if not os.path.exists(path):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        
if args.convert_csv:
    required_files = ['\\' + netw  + '.' for netw in node_networks] + ['\\labels.'] + ['\\edges_' + netw + '.' for netw in node_networks]
    required_files = [path + req + 'csv' for req in required_files]
    
    for req in required_files:
        if not os.path.exists(req):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), req)
    
    for req in required_files:
        if 'labels.csv' in req:
            labels = torch.flatten(torch.tensor(np.array(pd.read_csv(req))))
            with open(req[:-3] + 'pkl', 'wb') as f:
                pickle.dump(labels, f)   
        else:
            df = pd.read_csv(req, index_col=0)
            df.to_pickle(req[:-3] + "pkl")
    
    if os.path.exists(path + '\\mask_values.csv'):
        padded_mask = [np.array(n).astype(np.int32)[0] for n in np.matrix(pd.read_csv(path + '\\mask_values.csv').T)]
        na_values_bools = [str(b) != '-2147483648' for b in padded_mask[1]]
        padded_mask[1] = padded_mask[1][na_values_bools]
        with open(path + '\\mask_values.pkl', 'wb') as f:
            pickle.dump(padded_mask, f)



if  enable_CUDA and torch.cuda.is_available() and gpu_id == 0:
    device = torch.device('cuda')
elif (gpu_id != 0 and torch.cuda.is_available() and enable_CUDA):
    device = torch.device('cuda:' + str(gpu_id))
else:
    device = torch.device('cpu')


def train():
    if enable_CUDA and torch.cuda.is_available():
      model.cuda().train()
    else:
      model.train()
    optimizer.zero_grad()
    if enable_CUDA and torch.cuda.is_available():
      out, emb1 = model(data.cuda())
    else:
       out, emb1 = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return emb1


def validate():
    model.eval()
    with torch.no_grad():
        out, emb2 = model(data)
        pred = out.argmax(dim=1)
        loss = criterion(out[data.valid_mask], data.y[data.valid_mask])        
    return loss, emb2

criterion = torch.nn.CrossEntropyLoss()

data_path_node =  base_path + 'data/' + dataset_name +'/'
run_name = 'SUPREME_'+  dataset_name + '_results'
save_path = base_path + run_name + '/'
excel_file = save_path + "SUPREME_results.xlsx"

if not os.path.exists(base_path + run_name):
    os.makedirs(base_path + run_name + '/')

file = base_path + 'data/' + dataset_name +'/labels.pkl'
with open(file, 'rb') as f:
    labels = pickle.load(f)

file = base_path + 'data/' + dataset_name + '/mask_values.pkl'
if os.path.exists(file):
    with open(file, 'rb') as f:
        train_valid_idx, test_idx = pickle.load(f)
else:
    train_valid_idx, test_idx= train_test_split(np.arange(len(labels)), test_size=0.20, shuffle=True, stratify=labels, random_state=random_state)

start = time.time()

is_first = 0

for netw in node_networks:
    file = base_path + 'data/' + dataset_name +'/'+ netw +'.pkl'
    with open(file, 'rb') as f:
        feat = pickle.load(f)
        if feature_selection_per_network[node_networks.index(netw)] and top_features_per_network[node_networks.index(netw)] < feat.values.shape[1]:     
            print('SUPREME is running feature selection for input feature of ' + netw + '..')
            feat_flat = [item for sublist in feat.values.tolist() for item in sublist]
            feat_temp = robjects.FloatVector(feat_flat)
            robjects.globalenv['feat_matrix'] = robjects.r('matrix')(feat_temp)
            robjects.globalenv['feat_x'] = robjects.IntVector(feat.shape)
            robjects.globalenv['labels_vector'] = robjects.IntVector(labels.tolist())
            robjects.globalenv['top'] = top_features_per_network[node_networks.index(netw)]
            robjects.globalenv['maxBorutaRuns'] = boruta_runs
            robjects.r('''
                require(rFerns)
                require(Boruta)
                labels_vector = as.factor(labels_vector)
                feat_matrix <- Reshape(feat_matrix, feat_x[1])
                feat_data = data.frame(feat_matrix)
                colnames(feat_data) <- 1:feat_x[2]
                feat_data <- feat_data %>%
                    mutate('Labels' = labels_vector)
                boruta.train <- Boruta(feat_data$Labels ~ ., data= feat_data, doTrace = 0, getImp=getImpFerns, holdHistory = T, maxRuns = maxBorutaRuns)
                thr = sort(attStats(boruta.train)$medianImp, decreasing = T)[top]
                boruta_signif = rownames(attStats(boruta.train)[attStats(boruta.train)$medianImp >= thr,])
                    ''')
            boruta_signif = robjects.globalenv['boruta_signif']
            robjects.r.rm("feat_matrix")
            robjects.r.rm("labels_vector")
            robjects.r.rm("feat_data")
            robjects.r.rm("boruta_signif")
            robjects.r.rm("thr")
            topx = []
            for index in boruta_signif:
                t_index=re.sub("`","",index)
                topx.append((np.array(feat.values).T)[int(t_index)-1])
            topx = np.array(topx)
            values = torch.tensor(topx.T, device=device)
        elif feature_selection_per_network[node_networks.index(netw)] and top_features_per_network[node_networks.index(netw)] >= feat.values.shape[1]:
            print('SUPREME ignored feature selection for network ' + netw + ' since the defined number of top features is not less than the number of features.')
            values = feat.values
        else:
            values = feat.values
    
    if is_first == 0:
        new_x = torch.tensor(values, device=device).float()
        is_first = 1
    else:
        new_x = torch.cat((new_x, torch.tensor(values, device=device).float()), dim=1)
    
    print('SUPREME has ' + str(values.shape[1]) + ' features for ' + netw + '.')
    
for n in range(len(node_networks)):
    netw_base = node_networks[n]
    print('SUPREME is generating ' + netw_base + ' embedding..')
    with open(data_path_node + 'edges_' + netw_base + '.pkl', 'rb') as f:
        edge_index = pickle.load(f)
    best_ValidLoss = np.Inf

    for learning_rate in learning_rates:
        for hid_size in hid_sizes:
            av_valid_losses = list()

            for ii in range(xtimes2):
                data = Data(x=new_x, edge_index=torch.tensor(edge_index[edge_index.columns[0:2]].transpose().values, device=device).long(),
                            edge_attr=torch.tensor(edge_index[edge_index.columns[2]].transpose().values, device=device).float(), y=labels) 
                X = data.x[train_valid_idx]
                y = data.y[train_valid_idx]
                rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1)

                for train_part, valid_part in rskf.split(X, y):
                    train_idx = train_valid_idx[train_part]
                    valid_idx = train_valid_idx[valid_part]
                    break

                train_mask = np.array([i in set(train_idx) for i in range(data.x.shape[0])])
                valid_mask = np.array([i in set(valid_idx) for i in range(data.x.shape[0])])
                data.valid_mask = torch.tensor(valid_mask, device=device)
                data.train_mask = torch.tensor(train_mask, device=device)
                test_mask = np.array([i in set(test_idx) for i in range(data.x.shape[0])])
                data.test_mask = torch.tensor(test_mask, device=device)

                in_size = data.x.shape[1]
                out_size = torch.unique(data.y).shape[0]
                model = module.Net(in_size=in_size, hid_size=hid_size, out_size=out_size)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                min_valid_loss = np.Inf
                patience_count = 0

                for epoch in range(max_epochs):
                    emb = train()
                    this_valid_loss, emb = validate()

                    if this_valid_loss < min_valid_loss:
                        min_valid_loss = this_valid_loss
                        patience_count = 0
                    else:
                        patience_count += 1

                    if epoch >= min_epochs and patience_count >= patience:
                        break

                av_valid_losses.append(min_valid_loss.item())

            av_valid_loss = round(statistics.median(av_valid_losses), 3)
            
            if av_valid_loss < best_ValidLoss:
                best_ValidLoss = av_valid_loss
                best_emb_lr = learning_rate
                best_emb_hs = hid_size
                
    data = Data(x=new_x, edge_index=torch.tensor(edge_index[edge_index.columns[0:2]].transpose().values, device=device).long(),
                edge_attr=torch.tensor(edge_index[edge_index.columns[2]].transpose().values, device=device).float(), y=labels) 
    X = data.x[train_valid_idx]
    y = data.y[train_valid_idx]
    
    train_mask = np.array([i in set(train_valid_idx) for i in range(data.x.shape[0])])
    data.train_mask = torch.tensor(train_mask, device=device)
    valid_mask = np.array([i in set(test_idx) for i in range(data.x.shape[0])])
    data.valid_mask = torch.tensor(valid_mask, device=device)
    
    in_size = data.x.shape[1]
    out_size = torch.unique(data.y).shape[0]
    model = module.Net(in_size=in_size, hid_size=best_emb_hs, out_size=out_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_emb_lr)

    min_valid_loss = np.Inf
    patience_count = 0
                
    for epoch in range(max_epochs):
        emb = train()
        this_valid_loss, emb = validate()

        if this_valid_loss < min_valid_loss:
            min_valid_loss = this_valid_loss
            patience_count = 0
            selected_emb = emb
        else:
            patience_count += 1

        if epoch >= min_epochs and patience_count >= patience:
            break


    emb_file = save_path + 'Emb_' +  netw_base + '.pkl'
    with open(emb_file, 'wb') as f:
        pickle.dump(selected_emb, f)
        pd.DataFrame(selected_emb).to_csv(emb_file[:-4] + '.csv')
    
    print('Embedding for ' + netw_base + ' is generated with lr ' + str(best_emb_lr) + ' and hs ' + str(best_emb_hs) + '.')

print('SUPREME is integrating embeddings with ' + int_method + '..')
addFeatures = []
t = range(len(node_networks))
trial_combs = []
for r in range(1, len(t) + 1):
    trial_combs.extend([list(x) for x in itertools.combinations(t, r)])


for trials in range(len(trial_combs)):
    node_networks2 = [node_networks[i] for i in trial_combs[trials]] # list(set(a) & set(feature_networks))
    netw_base = node_networks2[0]
    emb_file = save_path + 'Emb_' +  netw_base + '.pkl'
    with open(emb_file, 'rb') as f:
        emb = pickle.load(f)

    if len(node_networks2) > 1:
        for netw_base in node_networks2[1:]:
            emb_file = save_path + 'Emb_' +  netw_base + '.pkl'
            with open(emb_file, 'rb') as f:
                cur_emb = pickle.load(f)
            emb = torch.cat((emb, cur_emb), dim=1)
            
    if addRawFeat == True:
        is_first = 0
        addFeatures = feature_networks_integration
        for netw in addFeatures:
            file = base_path + 'data/' + dataset_name +'/'+ netw +'.pkl'
            with open(file, 'rb') as f:
                feat = pickle.load(f)
            if is_first == 0:
                allx = torch.tensor(feat.values, device=device).float()
                is_first = 1
            else:
                allx = torch.cat((allx, torch.tensor(feat.values, device=device).float()), dim=1)   
        
        if optional_feat_selection == True:     
            print('SUPREME is running the optional feature selection for raw features to integrate..')
            allx_flat = [item for sublist in allx.tolist() for item in sublist]
            allx_temp = robjects.FloatVector(allx_flat)
            robjects.globalenv['allx_matrix'] = robjects.r('matrix')(allx_temp)
            robjects.globalenv['allx_x'] = robjects.IntVector(allx.shape)
            robjects.globalenv['labels_vector'] = robjects.IntVector(labels.tolist())
            robjects.globalenv['top'] = boruta_top_features
            robjects.globalenv['maxBorutaRuns'] = boruta_runs
            robjects.r('''
                require(rFerns)
                require(Boruta)
                labels_vector = as.factor(labels_vector)
                allx_matrix <- Reshape(allx_matrix, allx_x[1])
                allx_data = data.frame(allx_matrix)
                colnames(allx_data) <- 1:allx_x[2]
                allx_data <- allx_data %>%
                    mutate('Labels' = labels_vector)
                boruta.train <- Boruta(allx_data$Labels ~ ., data= allx_data, doTrace = 0, getImp=getImpFerns, holdHistory = T, maxRuns = maxBorutaRuns)
                thr = sort(attStats(boruta.train)$medianImp, decreasing = T)[top]
                boruta_signif = rownames(attStats(boruta.train)[attStats(boruta.train)$medianImp >= thr,])
                    ''')
            boruta_signif = robjects.globalenv['boruta_signif']
            robjects.r.rm("allx_matrix")
            robjects.r.rm("labels_vector")
            robjects.r.rm("allx_data")
            robjects.r.rm("boruta_signif")
            robjects.r.rm("thr")
            topx = []
            for index in boruta_signif:
                t_index=re.sub("`","",index)
                topx.append((np.array(allx).T)[int(t_index)-1])
            topx = np.array(topx)
            emb = torch.cat((emb, torch.tensor(topx.T, device=device)), dim=1)
            print('Top ' + str(boruta_top_features) + " features have been selected.")
        else:
            emb = torch.cat((emb, allx), dim=1)
    
    data = Data(x=emb, y=labels)
    train_mask = np.array([i in set(train_valid_idx) for i in range(data.x.shape[0])])
    data.train_mask = torch.tensor(train_mask, device=device)
    test_mask = np.array([i in set(test_idx) for i in range(data.x.shape[0])])
    data.test_mask = torch.tensor(test_mask, device=device)
    X_train = pd.DataFrame(data.x[data.train_mask].cpu().numpy())
    X_test = pd.DataFrame(data.x[data.test_mask].cpu().numpy())
    y_train = pd.DataFrame(data.y[data.train_mask].cpu().numpy()).values.ravel()
    y_test = pd.DataFrame(data.y[data.test_mask].cpu().numpy()).values.ravel()
    
    if int_method == 'MLP':
        params = {'hidden_layer_sizes': [(16,), (32,),(64,),(128,),(256,),(512,), (32, 32), (64, 32), (128, 32), (256, 32), (512, 32)],
                  'learning_rate_init': [0.1, 0.01, 0.001, 0.0001, 0.00001, 1, 2, 3],
                  'max_iter': [250, 500, 1000, 1500, 2000],
                  'n_iter_no_change': range(10,110,10)}
        search = RandomizedSearchCV(estimator = MLPClassifier(solver = 'adam', activation = 'relu', early_stopping = True), 
                                    return_train_score = True, scoring = 'f1_macro', 
                                    param_distributions = params, cv = 4, n_iter = xtimes, verbose = 0)
        search.fit(X_train, y_train)
        model = MLPClassifier(solver = 'adam', activation = 'relu', early_stopping = True,
                              max_iter = search.best_params_['max_iter'], 
                              n_iter_no_change = search.best_params_['n_iter_no_change'],
                              hidden_layer_sizes = search.best_params_['hidden_layer_sizes'],
                              learning_rate_init = search.best_params_['learning_rate_init'])
        
    elif int_method == 'XGBoost':
        params = {'reg_alpha':range(0,10,1), 'reg_lambda':range(1,10,1) ,'max_depth': range(1,6,1), 
                  'min_child_weight': range(1,10,1), 'gamma': range(0,6,1),
                  'learning_rate':[0, 1e-5, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
                  'max_delta_step': range(0,10,1), 'colsample_bytree': [0.5, 0.7, 1.0],
                  'colsample_bylevel': [0.5, 0.7, 1.0], 'colsample_bynode': [0.5, 0.7, 1.0]}
        fit_params = {'early_stopping_rounds': 10,
                     'eval_metric': 'mlogloss',
                     'eval_set': [(X_train, y_train)]}
        
        if enable_CUDA and torch.cuda.is_available(): 
            search = RandomizedSearchCV(estimator = XGBClassifier(use_label_encoder=False, n_estimators = 1000, 
                                                                  fit_params = fit_params, objective="multi:softprob", eval_metric = "mlogloss", 
                                                                  verbosity = 0, tree_method = 'gpu_hist', gpu_id=gpu_id), return_train_score = True, scoring = 'f1_macro',
                                        param_distributions = params, cv = 4, n_iter = xtimes, verbose = 0)
        else:          
            search = RandomizedSearchCV(estimator = XGBClassifier(use_label_encoder=False, n_estimators = 1000, 
                                                                  fit_params = fit_params, objective="multi:softprob", eval_metric = "mlogloss", 
                                                                  verbosity = 0), return_train_score = True, scoring = 'f1_macro',
                                        param_distributions = params, cv = 4, n_iter = xtimes, verbose = 0)
        
        search.fit(X_train, y_train)
        
        model = XGBClassifier(use_label_encoder=False, objective="multi:softprob", eval_metric = "mlogloss", verbosity = 0,
                              n_estimators = 1000, fit_params = fit_params,
                              reg_alpha = search.best_params_['reg_alpha'],
                              reg_lambda = search.best_params_['reg_lambda'],
                              max_depth = search.best_params_['max_depth'],
                              min_child_weight = search.best_params_['min_child_weight'],
                              gamma = search.best_params_['gamma'],
                              learning_rate = search.best_params_['learning_rate'],
                              max_delta_step = search.best_params_['max_delta_step'],
                              colsample_bytree = search.best_params_['colsample_bytree'],
                              colsample_bylevel = search.best_params_['colsample_bylevel'],
                              colsample_bynode = search.best_params_['colsample_bynode'])
                            
    elif int_method == 'RF':
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        params = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                  'max_depth': max_depth,
                  'min_samples_split': [2, 5, 7, 10],
                  'min_samples_leaf': [1, 2, 5, 7, 10], 
                 'min_impurity_decrease':[0,0.5, 0.7, 1, 5, 10],
                 'max_leaf_nodes': [None, 5, 10, 20]}
        search = RandomizedSearchCV(estimator = RandomForestClassifier(), return_train_score = True,
                                    scoring = 'f1_macro', param_distributions = params, cv=4,  n_iter = xtimes, verbose = 0)
        search.fit(X_train, y_train)
        model=RandomForestClassifier(n_estimators = search.best_params_['n_estimators'],
                                     max_depth = search.best_params_['max_depth'],
                                     min_samples_split = search.best_params_['min_samples_split'],
                                     min_samples_leaf = search.best_params_['min_samples_leaf'],
                                     min_impurity_decrease = search.best_params_['min_impurity_decrease'],
                                     max_leaf_nodes = search.best_params_['max_leaf_nodes'])

    elif int_method == 'SVM':
        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
                  'kernel': ['linear', 'rbf']}
        search = RandomizedSearchCV(SVC(), return_train_score = True,
                                    scoring = 'f1_macro', param_distributions = params, cv=4, n_iter = xtimes, verbose = 0)
        search.fit(X_train, y_train)
        model=SVC(kernel=search.best_params_['kernel'],
                  C = search.best_params_['C'],
                  gamma = search.best_params_['gamma'])

 
    av_result_acc = list()
    av_result_wf1 = list()
    av_result_mf1 = list()
    av_tr_result_acc = list()
    av_tr_result_wf1 = list()
    av_tr_result_mf1 = list()
 
        
    for ii in range(xtimes2):
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        y_pred = [round(value) for value in predictions]
        preds = model.predict(pd.DataFrame(data.x.cpu().numpy()))
        av_result_acc.append(round(accuracy_score(y_test, y_pred), 3))
        av_result_wf1.append(round(f1_score(y_test, y_pred, average='weighted'), 3))
        av_result_mf1.append(round(f1_score(y_test, y_pred, average='macro'), 3))
        tr_predictions = model.predict(X_train)
        tr_pred = [round(value) for value in tr_predictions]
        av_tr_result_acc.append(round(accuracy_score(y_train, tr_pred), 3))
        av_tr_result_wf1.append(round(f1_score(y_train, tr_pred, average='weighted'), 3))
        av_tr_result_mf1.append(round(f1_score(y_train, tr_pred, average='macro'), 3))
        
    if xtimes2 == 1:
        av_result_acc.append(round(accuracy_score(y_test, y_pred), 3))
        av_result_wf1.append(round(f1_score(y_test, y_pred, average='weighted'), 3))
        av_result_mf1.append(round(f1_score(y_test, y_pred, average='macro'), 3))
        av_tr_result_acc.append(round(accuracy_score(y_train, tr_pred), 3))
        av_tr_result_wf1.append(round(f1_score(y_train, tr_pred, average='weighted'), 3))
        av_tr_result_mf1.append(round(f1_score(y_train, tr_pred, average='macro'), 3))
        

    result_acc = str(round(statistics.median(av_result_acc), 3)) + '+-' + str(round(statistics.stdev(av_result_acc), 3))
    result_wf1 = str(round(statistics.median(av_result_wf1), 3)) + '+-' + str(round(statistics.stdev(av_result_wf1), 3))
    result_mf1 = str(round(statistics.median(av_result_mf1), 3)) + '+-' + str(round(statistics.stdev(av_result_mf1), 3))
    tr_result_acc = str(round(statistics.median(av_tr_result_acc), 3)) + '+-' + str(round(statistics.stdev(av_tr_result_acc), 3))
    tr_result_wf1 = str(round(statistics.median(av_tr_result_wf1), 3)) + '+-' + str(round(statistics.stdev(av_tr_result_wf1), 3))
    tr_result_mf1 = str(round(statistics.median(av_tr_result_mf1), 3)) + '+-' + str(round(statistics.stdev(av_tr_result_mf1), 3))
    
    
    df = pd.DataFrame(columns=['Comb No', 'Used Embeddings', 'Added Raw Features', 'Selected Params', 'Train Acc', 'Train wF1','Train mF1', 'Test Acc', 'Test wF1','Test mF1'])
    x = [trials, node_networks2, addFeatures, search.best_params_, 
         tr_result_acc, tr_result_wf1, tr_result_mf1, result_acc, result_wf1, result_mf1]
    df = df.append(pd.Series(x, index=df.columns), ignore_index=True)
    
    print('Combination ' + str(trials) + ' ' + str(node_networks2) + ' >  selected parameters = ' + str(search.best_params_) + 
      ', train accuracy = ' + str(tr_result_acc) + ', train weighted-f1 = ' + str(tr_result_wf1) +
      ', train macro-f1 = ' +str(tr_result_mf1) + ', test accuracy = ' + str(result_acc) + 
      ', test weighted-f1 = ' + str(result_wf1) +', test macro-f1 = ' +str(result_mf1))

    if trials == 0:
        if addRawFeat == True:
            function.append_df_to_excel(excel_file, df, sheet_name = int_method + '+Raw', index = False, header = True)
        else:
            function.append_df_to_excel(excel_file, df, sheet_name = int_method, index = False, header = True)
    else:
        if addRawFeat == True:
            function.append_df_to_excel(excel_file, df, sheet_name = int_method + '+Raw', index = False, header = False)
        else:
            function.append_df_to_excel(excel_file, df, sheet_name = int_method, index = False, header = False)

end = time.time()
print('It took ' + str(round(end - start, 1)) + ' seconds for all runs.')
print('SUPREME is done.')
print('Results are available at ' + excel_file)
