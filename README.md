# Counterfactual Explanation Generation for Multivariate Time Series Forecasting  

This repository accompanies our research paper, which addresses the challenge of explainability in deep learning models, particularly in the context of multivariate time series forecasting.  

## 🔍 **Overview**  
Deep learning models excel at predicting complex patterns in multivariate time series, making them invaluable in critical domains such as healthcare and finance. However, their "black-box" nature raises concerns about trust and interpretability. To address this, we propose two novel methods for generating counterfactual explanations:  
1. **GENO-TOPSIS**: Combines Genetic Algorithms with the TOPSIS multi-criteria decision-making approach.  
2. **NSGA-II**: A faster alternative that achieves comparable results.  

To ensure these explanations are valid within domain-specific constraints, we introduce:  
- **CEVO**: The Counterfactual Explanation Validation Ontology.  
- CEVO leverages SWRL rules and SPARQL queries for validation.  

We demonstrate the effectiveness of our approach through a case study on estimating the State of Charge (SoC) of LFP battery cells.  

---

## 🛠️ **Key Features**  
- **Explanation Generation**: GENO-TOPSIS and NSGA-II methods for counterfactual explanations.  
- **Ontology-Based Validation**: CEVO framework using domain-specific knowledge for robust validation.  
- **Case Study**: Application in battery management systems to estimate State of Charge (SoC).  

---

## 📂 **Repository Contents**  
- **`code/`**: Implementation of GENO-TOPSIS, NSGA-II, with step-by-step examples and results.  
- **`data/`**: Example datasets for testing and validation.  
- **`docs/`**: Documentation and resources related to the paper.
  
---

## 📄 Citing this Work
If you use this repository, please cite:
@article{arbaoui2024counterfactual,  
  title={Validation ontologique des explications contrefactuelles pour les séries temporelles : Application aux batteries lithium-ion}, 
  author={Slimane Arbaoui, Ali Ayadi, Ahmed Samet, Tedjani Mesbahi, Romuald Boné},  
  journal={Conférence Extraction et Gestion des Connaissances (EGC)},  
  year={2025}  
}

---
## 📧 Contact
For questions or collaboration opportunities, feel free to contact:
Slimane Arbaoui  
Email: slimane.arbaoui@insa-strasbourg.fr  

---
