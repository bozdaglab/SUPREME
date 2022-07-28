## SUPREME: A cancer subtype prediction methodology integrating multiomics data using Graph Convolutional Neural Network

 <img src="https://ziynetnesibe.com/wp-content/uploads/2022/07/Figure1-2.png" width="800" height="450" />
 <!-- ![SUPREME pipeline]  -->
 
SUPREME (a cancer `SU`btype `PRE`diction `ME`thodology)

An integrative node classification framework, called SUPREME (a **su**btype **pre**diction **me**thodology), that utilizes graph convolutions on multiple datatype-specific networks that are annotated with multiomics datasets as node features. This framework is model-agnostic and could be applied to any classification problem with properly processed datatypes and networks. In our work, SUPREME was applied specifically to the breast cancer subtype prediction problem by applying convolution on patient similarity networks constructed based on multiple biological datasets from breast tumor samples.

First, SUPREME generates network-specific patient embeddings from each datatype separately. Then using those embedding, it does cancer subtype prediction through all the combinations of embeddings, and report the evaluation results.

---

## How to run SUPREME?

Use `SUPREME.py` to run SUPREME.
Parameters:
`-csv` : It will automatically convert '.csv' files to '.pkl' files and save them for each input datatype given as `node_networks`.
`-data`: It specifies the data location to use under the 'data' folder (default is 'sample data').
`-gpu`: It enables GPU support (default is False).
`-gpu_id`: It specifies the index of GPU device for users with multiple GPUS. (default is 0).

Example runs:
- `python SUPREME.py`: runs SUPREME using pkl data files under 'data/sample_data' folder
- `python SUPREME.py -csv`:  runs SUPREME using csv data files under 'data/sample_data' folder
- `python SUPREME.py -data user_defined_data`:  runs SUPREME using pkl data files under 'data/user_defined_data' folder
- `python SUPREME.py -csv -data user_defined_data`:  runs SUPREME using csv data files under 'data/user_defined_data' folder
- `python SUPREME.py -gpu`:  runs SUPREME using pkl data files under 'data/sample_data' folder with GPU support

Sample console output:
``` > python SUPREME.py -csv
SUPREME is setting up!
SUPREME has 10 features for clinical.
SUPREME has 250 features for cna.
SUPREME has 250 features for exp.
SUPREME is generating clinical embedding..
Embedding for clinical is generated with lr 0.001 and hs 512.
SUPREME is generating cna embedding..
Embedding for cna is generated with lr 0.001 and hs 256.
SUPREME is generating exp embedding..
Embedding for exp is generated with lr 0.001 and hs 128.
SUPREME is integrating embeddings with MLP..
Combination 0 ['clinical'] >  selected parameters = {'n_iter_no_change': 10, 'max_iter': 500, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (32, 32)}, train accuracy = 0.899+-0.067, train weighted-f1 = 0.889+-0.072, train macro-f1 = 0.845+-0.136, test accuracy = 0.815+-0.029, test weighted-f1 = 0.782+-0.034, test macro-f1 = 0.646+-0.08
Combination 1 ['cna'] >  selected parameters = {'n_iter_no_change': 80, 'max_iter': 2000, 'learning_rate_init': 0.1, 'hidden_layer_sizes': (32,)}, train accuracy = 0.935+-0.056, train weighted-f1 = 0.936+-0.062, train macro-f1 = 0.899+-0.115, test accuracy = 0.838+-0.04, test weighted-f1 = 0.839+-0.058, test macro-f1 = 0.734+-0.121
Combination 2 ['exp'] >  selected parameters = {'n_iter_no_change': 50, 'max_iter': 500, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (128,)}, train accuracy = 0.972+-0.057, train weighted-f1 = 0.972+-0.062, train macro-f1 = 0.962+-0.11, test accuracy = 0.846+-0.042, test weighted-f1 = 0.841+-0.048, test macro-f1 = 0.777+-0.083
Combination 3 ['clinical', 'cna'] >  selected parameters = {'n_iter_no_change': 90, 'max_iter': 2000, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (128,)}, train accuracy = 0.938+-0.048, train weighted-f1 = 0.939+-0.048, train macro-f1 = 0.919+-0.08, test accuracy = 0.838+-0.034, test weighted-f1 = 0.833+-0.039, test macro-f1 = 0.731+-0.105
Combination 4 ['clinical', 'exp'] >  selected parameters = {'n_iter_no_change': 70, 'max_iter': 250, 'learning_rate_init': 0.1, 'hidden_layer_sizes': (128,)}, train accuracy = 0.914+-0.051, train weighted-f1 = 0.915+-0.054, train macro-f1 = 0.847+-0.084, test accuracy = 0.816+-0.038, test weighted-f1 = 0.795+-0.037, test macro-f1 = 0.685+-0.072
Combination 5 ['cna', 'exp'] >  selected parameters = {'n_iter_no_change': 60, 'max_iter': 250, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (512,)}, train accuracy = 0.953+-0.032, train weighted-f1 = 0.954+-0.032, train macro-f1 = 0.941+-0.04, test accuracy = 0.831+-0.018, test weighted-f1 = 0.824+-0.017, test macro-f1 = 0.748+-0.038
Combination 6 ['clinical', 'cna', 'exp'] >  selected parameters = {'n_iter_no_change': 80, 'max_iter': 250, 'learning_rate_init': 0.001, 'hidden_layer_sizes': (64,)}, train accuracy = 0.972+-0.069, train weighted-f1 = 0.97+-0.072, train macro-f1 = 0.944+-0.123, test accuracy = 0.831+-0.038, test weighted-f1 = 0.814+-0.051, test macro-f1 = 0.652+-0.117
It took 1097.5 seconds for all runs.
SUPREME is done.
Results are available at SUPREME_sample_data_results/SUPREME_results.xlsx
```

### Input files: 
Files under the *sample_data* folder under *data* folder: 
- `labels.csv`: Labels of ordered samples (*i*th row has the label of sample with index *i*). First column is label starting from 0 till {number of subtype}-1. First row contains column name.
- Input features: *i*th row has the feature values of sample with index *i*. (Still, we have column names and row names, even not considered.)
  - `clinical.csv`: 257 Samples (row) x 10 normalized clinical features (column)
  - `cna.csv`: 257 Samples (row) x 250 normalized copy number aberration features (column)
  - `exp.csv`: 257 Samples (row) x 250 normalized gene expression features (column)

- Input networks: First column is rownames, second and third columns will contain sample indexes for the sample-sample pairs having interactions and forth column will be the weight of the interaction.
  - `edges_clinical.csv`: Clinical-based patient similarity network 
  - `edges_cna.csv`: Copy number aberration-based patient similarity network
  - `edges_exp.csv`: Gene expression-based patient similarity network

### Output files:
Files under the *SUPREME_sample_data_results* folder:
- `Emb_clinical.csv`: Clinical-based patient embedding
- `Emb_cna.csv`: Copy number aberration-based patient embedding
- `Emb_exp.csv`: Gene expression-based patient embedding
- `SUPREME_results.xlsx`: Evaluation results for each embedding combination. It contains selected hyperparameters and evaluation metrics (accuracy, weighted F1, and macro F1 scores for both the training and testing data) for each embedding combination.
---

## How to customize SUPREME?

Files under *lib* folder:
- `function.py`: Includes functions.
- `module.py`: Graph Convolutional Neural Network-related module.

SUPREME Flowchart
 <img src="[https://ziynetnesibe.com/wp-content/uploads/2022/07/SUPREME_Flowchart-1024x264.png](https://ziynetnesibe.com/wp-content/uploads/2022/07/SUPREME_Flowchart-1024x264.png)" width="800" height="450" />
 <!-- ![SUPREME Flowchart]  -->
 
### User Options

- Adjust the following variables (lines 2-7):
  - `addRawFeat`: *True* or *False*: If *True*, raw features from listed datatypes in `features_to_integrate` will be integrated during prediction; if *False*, no raw features will be integrated (default is *True*). 
  - `base_path`: the path to SUPREME github folder
  - `dataset_name`: the data folder name in `base_path` including required input data to run SUPREME
  - `feature_networks_integration`: list of the datatypes to integrate as raw features
  - `node_networks`: list of the datatypes to use (should have at least one datatype)
  - `int_method`: method to integrate during the prediction of subtypes. Options are 'MLP' for Multi-layer Perceptron, 'XGBoost' for XGBoost, 'RF' for Random Forest, 'SVM' for Support Vector Machine. (default is 'MLP'.)
  - `feature_selection_per_network`: a list of *True* or *False*: If *True*, the corresponding `top_features_per_network` features are selected from feature selection algorithm; if *False*, all features are used for integration. (order of `feature_selection_per_network` and `top_features_per_network` are same as order of `node_networks`)
  - `top_features_per_network`: list of numbers: If corresponding `feature_selection_per_network` is *True* and corresponding `top_features_per_network` is less than the input feature number, then feature selection algorithm will be applied for that network. (order of `feature_selection_per_network` and `top_features_per_network` are same as order of `node_networks`)
  - `boruta_top_features`: the number of top raw features to be integrated as raw features if `optional_feat_selection` and `addRawFeat` are *True*; otherwise ignored.
  - `optional_feat_selection`: *True* or *False*: If *True*, the top `boruta_top_features` features from each combination of integrated networks are added as raw features; if *False*, all the raw features are added to the embedding. (considered only if `addRawFeat` is *True*)
  
- Adjust the following hyperparameters (lines 8-15):
  - `max_epochs`: maximum number of epoch (default is 500.)
  - `min_epochs`: minimum number of epoch (default is 200.)
  - `patience`: patience for early stopping (default is 30.)
  - `learning_rates`: list of values to try as learning rate (default is [0.001, 0.01, 0.1].)
  - `hid_sizes`: list of values to try as hidden layer size (default is [16, 32, 64, 128, 256, 512].)
  - `xtimes`: the number of SUPREME runs to select the best hyperparameter combination during hyperparameter tuning as part of Randomized Search (default: 50, should be more than 1.)
  - `xtimes2`: the number of SUPREME runs for the selected hyperparameter combination, used to generate the median statistics (default: 10, should be more than 1.) 
  - `boruta_runs`: the number of times Boruta runs to determine feature significance (default: 100, should be more than 1) (considered only if `addRawFeat` and `optional_feat_selection` are *True*, or if any of the values in `feature_selection_per_network` are *True*)
  - `enable_CUDA`: *True* or *False*: Enables CUDA if *True* (default is *False*).
  - `gpu_id`: For users with multiple GPUs, this specifies the index of the GPU device to use (default is 0.)

---

### Data Generation for a New Dataset
- `base_path` should contain a folder named `dataset_name` (called as *data folder* afterwards) under `data` folder . 
- `node_networks` will have the list of the datatype names that will be used for SUPREME run. These names are user-defined, but should be consistent for all the file names.
- In the *data folder*, there should be one label file named `labels.pkl`. 
  - `labels.pkl`: *<class 'torch.Tensor'>* with the shape of *torch.Size([{*sample size*}])*
- In addition, the *data folder* will contain two '.pkl files per datatype. 
  - `{datatype name}.pkl`: *<class 'pandas.core.frame.DataFrame'>* with the shape of *({sample size}, {selected feature size for that datatype})*
  - `edges_{datatype name}.pkl`: *<class 'pandas.core.frame.DataFrame'>* with the shape of *({Number of patient-patient pair interaction for this datatype}, 3)*. First and second columns will contain patient indexes for the patient-patient pairs having interactions and third column will be the weight of the interaction.
- The *data folder* might have a file named `mask_values.pkl` *(<class 'list'>)* if the user wants to specify test samples. If `mask_values.pkl` does not exist in *data folder*, SUPREME will generate train and test splits. If added, `mask_values.pkl` needs to have two variables in it:
  - `train_valid_idx`: *<class 'numpy.ndarray'>* with the shape of *({Number of samples for training and validation,)* containing the sample indexes for training and validation.
  - `test_idx`: *<class 'numpy.ndarray'>* with the shape of *({Number of samples for test,)* containing the sample indexes for test.
 
 

***!! Note that*** sample size and the order of the samples should be the same for whole variables. Sample indexes should start from 0 till *sample size-1* consistent with the sample order.  
- `labels.pkl` will have the labels of the ordered samples. (*i*th value has the label of sample with index *i*)  
- `{datatype name}.pkl` will have the values of the ordered samples in each datatype (feature size could be datatype specific). (*i*th row has the feature values of sample with index *i*)  
- `edges_{datatype name}.pkl` will have the matching sample indexes to represent interactions.  
- `train_valid_idx` and `test_idx` will contain the matching sample indexes.


 
