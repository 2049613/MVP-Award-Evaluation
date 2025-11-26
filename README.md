# MVP Award Evaluation — Unsupervised Ranking & Rule-Based Models

This repository implements an **unsupervised and interpretable framework** for evaluating NBA MVP candidates using player-season statistical data. It focuses on algorithm code for:

- Unsupervised feature ranking  
- Player scoring based on weighted statistics  
- Rule-based explainable machine learning (RuleFit & Skope-Rules)  
- MVP / Top-3 prediction using statistical patterns only  

The repository provides runnable scripts that can be directly executed to generate rankings, rules, and predictions.

---

## Unsupervised Feature-weighted Ensemble Ranking Algorithm

### **1. Feature Importance Evaluation Algorithm** (ensemble_ranking.py)
The project includes implementations of three ranking methods:

- **Unsupervised Random Forest (URF)**
- **Genie3 (adapted for unsupervised use)**
- **URelief**

All three outputs are combined using an **ensemble weighting strategy**. These methods generate feature-importance scores based solely on statistical patterns, without labels.


### **2. Player Score Calculation** (ufer.py)
Based on feature importance, each player receives a season-level score, enabling:

- Unsupervised MVP prediction (Top-1)
- Unsupervised Top-3 candidate identification


## Unsupervised Player Evaluation

#### RuleFit (RuleFit.py)
- Produces interpretable rules + linear terms  
- Outputs MVP probability    

#### Skope-Rules (SkopeRules.py)
- Generates precision-filtered rules  
- Produces binary MVP / non-MVP decisions  

Both models use the same statistical features and can be trained on different label sets.

---


