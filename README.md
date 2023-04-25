## SUPREME: A cancer subtype prediction methodology integrating multiomics data using Graph Convolutional Neural Network

To learn more about SUPREME, read our paper at: https://www.biorxiv.org/content/10.1101/2022.08.03.502682v1

<img src="https://ziynetnesibe.com/wp-content/uploads/2022/07/SUPREME.png"  width="819" height="439"/>
 <!-- ![SUPREME pipeline]  -->
 
SUPREME (a cancer `SU`btype `PRE`diction `ME`thodology)

An integrative node classification framework, called SUPREME (a **su**btype **pre**diction **me**thodology), that utilizes graph convolutions on multiple datatype-specific networks that are annotated with multiomics datasets as node features. This framework is model-agnostic and could be applied to any classification problem with properly processed datatypes and networks. In our work, SUPREME was applied specifically to the breast cancer subtype prediction problem by applying convolution on patient similarity networks constructed based on multiple biological datasets from breast tumor samples.

First, SUPREME generates network-specific patient embeddings from each datatype separately. Then using those embedding, it does cancer subtype prediction through all the combinations of embeddings, and report the evaluation results.

---

## How to run SUPREME?

Use `SUPREME.py` to run SUPREME.
Parameter: `-data`: It specifies the data location to use under the 'data' folder (default is 'sample data').

Example runs:
- `python SUPREME.py`: runs SUPREME under 'data/sample_data' folder
- `python SUPREME.py -data user_defined_data`:  runs SUPREME under 'data/user_defined_data' folder

Sample console output:
``` > python SUPREME.py
SUPREME is setting up!
SUPREME is running..
It took 488.5 seconds for node embedding generation (12 trials for 3 seperate GCNs).
SUPREME is integrating the embeddings..
Combination 0 ['clinical'] >  selected parameters = {'hidden_layer_sizes': (256,)}, train accuracy = 0.948+-0.088, train weighted-f1 = 0.946+-0.092, train macro-f1 = 0.929+-0.157, test accuracy = 0.808+-0.047, test weighted-f1 = 0.782+-0.052, test macro-f1 = 0.571+-0.09
Combination 1 ['cna'] >  selected parameters = {'hidden_layer_sizes': (64, 32)}, train accuracy = 0.95+-0.084, train weighted-f1 = 0.949+-0.09, train macro-f1 = 0.917+-0.153, test accuracy = 0.815+-0.074, test weighted-f1 = 0.793+-0.091, test macro-f1 = 0.603+-0.143
Combination 2 ['exp'] >  selected parameters = {'hidden_layer_sizes': (256,)}, train accuracy = 0.935+-0.067, train weighted-f1 = 0.934+-0.073, train macro-f1 = 0.9+-0.136, test accuracy = 0.815+-0.026, test weighted-f1 = 0.798+-0.032, test macro-f1 = 0.609+-0.06
Combination 3 ['clinical', 'cna'] >  selected parameters = {'hidden_layer_sizes': (512, 32)}, train accuracy = 0.987+-0.052, train weighted-f1 = 0.986+-0.056, train macro-f1 = 0.986+-0.107, test accuracy = 0.846+-0.035, test weighted-f1 = 0.829+-0.039, test macro-f1 = 0.674+-0.09
Combination 4 ['clinical', 'exp'] >  selected parameters = {'hidden_layer_sizes': (128,)}, train accuracy = 0.93+-0.099, train weighted-f1 = 0.927+-0.11, train macro-f1 = 0.883+-0.187, test accuracy = 0.808+-0.051, test weighted-f1 = 0.784+-0.051, test macro-f1 = 0.598+-0.099
Combination 5 ['cna', 'exp'] >  selected parameters = {'hidden_layer_sizes': (32,)}, train accuracy = 0.943+-0.066, train weighted-f1 = 0.943+-0.074, train macro-f1 = 0.912+-0.133, test accuracy = 0.831+-0.031, test weighted-f1 = 0.825+-0.037, test macro-f1 = 0.702+-0.095
Combination 6 ['clinical', 'cna', 'exp'] >  selected parameters = {'hidden_layer_sizes': (256,)}, train accuracy = 0.979+-0.034, train weighted-f1 = 0.979+-0.035, train macro-f1 = 0.974+-0.06, test accuracy = 0.846+-0.033, test weighted-f1 = 0.836+-0.041, test macro-f1 = 0.718+-0.094
It took 532.6 seconds in total.
SUPREME is done.
```

### Input files: 
Files under the *sample_data* folder under *data* folder: 
- `labels.pkl`: Labels of ordered samples (*i*th row has the label of sample with index *i*). First column is label starting from 0 till {number of subtype}-1. First row contains column name.
- Input features: *i*th row has the feature values of sample with index *i*. (Still, we have column names and row names, even not considered.)
  - `clinical.pkl`: 257 Samples (row) x 10 normalized clinical features (column)
  - `cna.pkl`: 257 Samples (row) x 250 normalized copy number aberration features (column)
  - `exp.pkl`: 257 Samples (row) x 250 normalized gene expression features (column)

- Input networks: First column is rownames, second and third columns will contain sample indexes for the sample-sample pairs having interactions and forth column will be the weight of the interaction.
  - `edges_clinical.pkl`: Clinical-based patient similarity network 
  - `edges_cna.pkl`: Copy number aberration-based patient similarity network
  - `edges_exp.pkl`: Gene expression-based patient similarity network

### Output files:
Files under the *SUPREME_sample_data_results* folder:
- `Emb_clinical.csv`: Clinical-based patient embedding
- `Emb_cna.csv`: Copy number aberration-based patient embedding
- `Emb_exp.csv`: Gene expression-based patient embedding

### Files under *lib* folder:
- `module.py`: Graph Convolutional Neural Network-related module.
---

## How to customize SUPREME?

### SUPREME Flowchart
 <img src="https://ziynetnesibe.com/wp-content/uploads/2022/07/SUPREME_Flowchart.png"/>
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
  - `learning_rate`: learning rate (default is 0.001.)
  - `hidden_size`: hidden size (default is 256.)
  - `xtimes`: the number of SUPREME runs to select the best hyperparameter combination during hyperparameter tuning as part of Randomized Search (default: 50, should be more than 1.)
  - `xtimes2`: the number of SUPREME runs for the selected hyperparameter combination, used to generate the median statistics (default: 10, should be more than 1.) 
  - `boruta_runs`: the number of times Boruta runs to determine feature significance (default: 100, should be more than 1) (considered only if `addRawFeat` and `optional_feat_selection` are *True*, or if any of the values in `feature_selection_per_network` are *True*)
 
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



Relevant package versions in the environment:
```
# Name                    Version                   Build  Channel
cpuonly                   2.0                           0    pytorch
numpy                     1.19.2           py36hadc3359_0
pandas                    1.1.5                    pypi_0    pypi
pickle5                   0.0.12                   pypi_0    pypi
pip                       21.3.1                   pypi_0    pypi
python                    3.6.13               h3758d61_0
python-dateutil           2.8.2                    pypi_0    pypi
pytorch                   1.10.2              py3.6_cpu_0    pytorch
pytorch-mutex             1.0                         cpu    pytorch
rpy2                      3.4.5                    pypi_0    pypi
scikit-learn              0.24.2                   pypi_0    pypi
torch-geometric           2.0.3                    pypi_0    pypi
torch-scatter             2.0.9                    pypi_0    pypi
torch-sparse              0.6.12                   pypi_0    pypi
torchaudio                0.10.2                 py36_cpu  [cpuonly]  pytorch
torchvision               0.11.3                 py36_cpu  [cpuonly]  pytorch
xgboost                   1.5.2                    pypi_0    pypi
```
