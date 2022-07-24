# SUPREME (A SUbtype PREdiction MEthodology)
SUPREME: A Graph Convolutional Neural Network (GCN)-based approach for cancer subtype prediction

---

## How to run SUPREME?

Run `SUPREME.py` after generating the proper input data.

### User Options

- Adjust the following variables (lines 2-7):
  - `addRawFeat`: *True* or *False*: If *True*, raw features from listed datatypes in `features_to_integrate` will be integrated during prediction; if *False*, no raw features will be integrated (default is *True*). 
  - `base_path`: the path to SUPREME github folder
  - `dataset_name`: the data folder name in `base_path` including required input data to run SUPREME
  - `feature_networks_integration`: list of the datatypes to integrate as raw features
  - `node_networks`: list of the datatypes to use (should have at least one datatype)
  - `int_method`: method to integrate during the prediction of subtypes. Options are 'MLP' for Multi-layer Perceptron, 'XGBoost' for XGBoost, 'RF' for Random Forest, 'SVM' for Support Vector Machine. (default is 'MLP'.)
  - `feature_selection_per_network`: a list of *True* or *False*: If *True*, the corresponding `top_features_per_network` features are selected from feature selection algorithm. (order of `feature_selection_per_network` and `top_features_per_network` are same as order of `node_networks`)
  - `top_features_per_network`: list of numbers: If corresponding `feature_selection_per_network` is *True* and corresponding `top_features_per_network` is less than the input feature number, then feature selection algorithm will be applied for that network. (order of `feature_selection_per_network` and `top_features_per_network` are same as order of `node_networks`)
  - `optional_feat_selection`: *True* or *False*: If *True*, the top `boruta_top_features` are added as raw features; if *False*, all the raw features are added. (considered only if `addRawFeat` is *True*)
  
- Adjust the following hyperparameters (lines 8-15):
  - `max_epochs`: maximum number of epoch (default is 500.)
  - `min_epochs`: minimum number of epoch (default is 200.)
  - `patience`: patience for early stopping (default is 30.)
  - `learning_rates`: list of values to try as learning rate (default is [0.001, 0.01, 0.1].)
  - `hid_sizes`: list of values to try as hidden layer size (default is [16, 32, 64, 128, 256, 512].)
  - `xtimes`: the number of SUPREME runs to select hyperparameter combination (default: 50, should be more than 1.)
  - `xtimes2`: the number of SUPREME runs for the selected hyperparameter combination (default: 10, should be more than 1.) 
  - `boruta_runs`: the number of Boruta runs to determine feature significance (default: 100, should be more than 1) (considered only if `addRawFeat` and `optional_feat_selection` are *True*)
  - `enable_CUDA`: *True* or *False*: Enables CUDA if *True*.
  - `gpu_id`: default is 0.

---

### Data Generation for a New Dataset
- `base_path` should contain a folder named `dataset_name` (called as *data folder* afterwards) under `data` folder . 
- `node_networks` will have the list of the datatype names that will be used for SUPREME run. These names are user-defined, but should be consistent for all the file names.
- In the *data folder*, there should be one label file named `labels.pkl`. 
  - `labels.pkl`: *<class 'torch.Tensor'>* with the shape of *torch.Size([{*sample size*}])*
- In addition, the *data folder* will contain two '.pkl files per datatype. 
  - `{datatype name}.pkl`: *<class 'pandas.core.frame.DataFrame'>* with the shape of *({sample size}, {selected feature size for that datatype})*
  - `edges_{datatype name}.pkl`: *<class 'pandas.core.frame.DataFrame'>* with the shape of *({Number of patient-patient pair interaction for this datatype}, 3)*. First and second columns will contain patient indexes for the patient-patient pairs having interactions and third column will be the weight of the interaction.
- The *data folder* might have a file named `mask_values.pkl` if the user wants to specify test samples. `mask_values.pkl` will have two variables in it:
  - `train_valid_idx`: *<class 'numpy.ndarray'>* with the shape of *({Number of sample for training and validation,)* containing the sample indexes for training and validation.
  - `test_idx`: *<class 'numpy.ndarray'>* with the shape of *({Number of sample for test,)* containing the sample indexes for test.
 
 If `mask_values.pkl` does not exist in *data folder*, SUPREME will generate train and test splits.

***!! Note that*** sample size and the order of the samples should be the same for whole variables. Sample indexes should start from 0 till *sample size-1* consistent with the sample order.  
- `labels.pkl` will have the labels of the ordered samples. (*i*th value has the label of sample with index *i*)  
- `{datatype name}.pkl` will have the values of the ordered samples in each datatype (feature size could be datatype specific). (*i*th row has the feature values of sample with index *i*)  
- `edges_{datatype name}.pkl` will have the matching sample indexes to represent interactions.  
- `train_valid_idx` and `test_idx` will contain the matching sample indexes.
