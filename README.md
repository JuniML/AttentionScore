# AttentionScore

<p align="center">
  <img src="img/figure-1.png" alt="AttentionScore Logo" width="400"/>
</p>

<p align="center">
  <b>A Deep Learning–Based Target-Specific Scoring Function for METTL3 Virtual Screening</b>
</p>

---

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Conda](https://img.shields.io/badge/Conda-Environment-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)

---

## 📑 Table of Contents
- [Description](#description)
- [Features](#features)
- [Workflow](#workflow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Examples](#examples)
- [Run Prediction](#run-prediction)
- [Citation](#citation)
- [License](#license)

---

## 🧬 Description

**AttentionScore** is a **deep learning–based scoring function** designed for **structure-based virtual screening (SBVS)** of **METTL3**, a key RNA methyltransferase and emerging anticancer target.  

The framework integrates **multi-head attention** and **autoencoder-based latent compression** with ligand-centric and interaction-aware descriptors (ECFP4, Avalon, PLEC) to improve accuracy and reduce dataset biases.

---

## ✨ Features
- ⚡ **End-to-end pipeline**: from molecule preparation to prediction  
- 🧪 **Target-specific scoring** for METTL3  
- 🧠 **Attention + Autoencoder fusion** for robust representation learning  
- 🔬 **Bias-aware decoy generation** using [DeepCoy](https://github.com/AngelRuizMoreno/Jupyter_Dock)  
- 📊 **Feature engineering** with ODDT & RDKit (PLEC, ECFP4, Avalon)  
- 🖥️ **User-friendly Jupyter notebooks** for training and prediction  

---

## 🔄 Workflow
The typical workflow for AttentionScore involves:

1. **Retrieval of Molecules**  
   > Example notebook available in `Notebooks/`

2. **Generation of DeepCoy Decoys**  
   > ~100 decoys per active → 50 optimized decoys retained  

3. **SMILES to MOL2 Conversion**

4. **Molecular Docking**  
   > Performed using **smina**  

5. **Feature Generation**  
   > PLEC, ECFP4, Avalon (via ODDT & RDKit)  

6. **Model Training**  
   > Deep neural network with **multi-head attention** + **autoencoder**

7. **Prediction**  
   > User-friendly notebook for applying AttentionScore to your own molecules  

---

## ⚙️ Requirements
All dependencies are listed in `requirements.yml`.  

```bash
conda env create -f requirements.yml
conda activate DeepMETLL3
