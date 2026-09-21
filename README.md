#  Avalanche Accident Prediction

## Overview

This project was developed during my internship at the
**Laboratoire de Probabilités, Statistique et Modélisation (LPSM)**
at **Sorbonne Université**.

The objective was to study and model avalanche-related ski accidents
using statistical learning and geographical data.

The project combines data collection, geospatial processing and
statistical modeling to estimate accident probabilities according
to environmental characteristics.

---

##  Objectives

The main objectives were:

- Collect avalanche accident data
- Integrate geographical and environmental information
- Process spatial data
- Build an accident / non-accident dataset
- Model accident probability using Generalized Linear Models
- Evaluate predictive performance

---

##  Data

The project combined several sources of information, including:

- Avalanche accident records
- Ski touring geographical data
- Avalanche risk information
- Altitude and spatial information

Data were collected and integrated from multiple sources before
statistical modeling.

> Original datasets are not necessarily distributed in this
> repository. Public or synthetic examples may be used instead.

---

##  Geospatial Processing

Geographical preprocessing was performed to associate accident
locations with ski touring trajectories.

This included spatial distance calculations and projection of
accident locations onto nearby trajectories.

---

##  Statistical Modeling

The probability of an avalanche-related accident was modeled using
a **Generalized Linear Model (GLM)** with a logistic formulation.

The modeling workflow included:

1. Data preparation
2. Feature construction
3. Train/test preparation
4. Logistic regression
5. Probability estimation
6. Model evaluation

---

##  Model Evaluation

Predictive performance was evaluated using:

- ROC curve
- Area Under the Curve (AUC)
- Predicted probabilities
- Comparison between accidents and non-accidents

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Geospatial data processing
- APIs
- Matplotlib

---

##  Repository Structure

```text
avalanche-accident-prediction/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── data_collection.py
│   ├── geospatial_processing.py
│   ├── preprocessing.py
│   ├── modeling.py
│   └── evaluation.py
│
├── notebooks/
├── figures/
└── data/
    └── README.md
