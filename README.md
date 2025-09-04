# AttentionScore

![logo](img/figure-1.png)

## Table of content

- [**Description**](#description)

- [**Requirements**](#requirements)

- [**Installation**](#installation)

- [**Examples**](#examples)

- [**Run Prediction**](#Run_Prediction)

- [**Citation**](#citation)

- [**License**](#license) 


## Description

**AttentionScore is deep learning based scoring function for METTL3 structure based virtual screening.** <br><br>

The User have to through the following steps:

**1. Retrieval of Molecules**
> The notebook is present in the Notebook directory. 

**2. Convert smiles to mol2**
> The generated smiles for decoys and actives should be converted to mol2 file

**3. Molecular docking**
> Molecular docking was carried out using smina 

**4. Genrate Voxel features**
> RdkitGridFeaturizer from deepchem was used to convert docked complexes into voxel features; https://deepchem.io/

**5. Train model**
> In this study 3DCNN with mulihead attention was used. 


**6. Predict**
> A user-friendly jupyternotebook is prepared for users to use for their molecules
## Requirements
> The required libraries are present in the requirments.yml file.
## Installation
> Users have to use the following command to create a virtual environment for this project
```
conda env create -f requirments.yml
conda activate DeepCGASPred
```
## Examples
> Toy dataset are present in the example directory. the example.ipynb can be used .

## Run_Prediction
> To run the prediction, Use Jupyter Noteboob or the following command. The DeepCGASPred.py is in the Streamlit directory 
 ```
conda activate DeepCGASPred
DeepCGASPred.py -r receptor.pdb -l ligand.sdf -o output.csv
```
## Streamlit_app
> We have developed the graphical userinterface for DeepCGASPred using streamlit app. Install all necassary libraries and then run the following command to run the app.
 ```
conda activate DeepCGASPred
streamlit run DeepCGASPred-streamlit.py
```
![logo](img/comments.png)
## Citation
## License
> These notebooks are under MIT, see the LICENSE file for details
Question about usage or troubleshooting? Please leave a comment here
