# Building a Neural Network for Diabetes Risk Prediction
# Diabetes Risk Prediction using Deep Learning

## 📌 Project Overview
This project implements an end-to-end **neural network–based diabetes risk prediction system** using real-world public health survey data from the **CDC Diabetes Health Indicators dataset**.  

The goal is to simulate a **clinical pre‑screening tool** that helps healthcare providers identify **high‑risk patients** for further diagnostic testing, optimizing medical resources while prioritizing patient safety.

The solution is built with **PyTorch**, follows best practices in **data preprocessing, model evaluation, and experiment tracking**, and emphasizes **clinical relevance over raw accuracy**, making it suitable for healthcare-oriented machine learning applications.

---

## 🏥 Real‑World Context
Early identification of diabetes significantly reduces long‑term complications and healthcare costs.  
Hospitals often face limited capacity for comprehensive testing, requiring efficient pre‑screening mechanisms.

This model addresses that need by predicting diabetes risk based on **health, lifestyle, and demographic indicators**, acting as a decision‑support system rather than a standalone diagnostic tool.

---

## 🎯 Project Mission

**Company:** UdaciHealth  
**Client:** Regional hospital network  

**Problem**  
Limited resources to perform comprehensive diabetes testing across thousands of patients each month.

**Solution**  
An automated deep learning pre‑screening system that prioritizes patients for diagnostic follow‑up based on predicted diabetes risk.

**Success Criteria**
- High **recall** to minimize missed diabetic cases
- Acceptable **precision** to limit unnecessary follow-up tests
- Clinically interpretable evaluation and deployment‑ready workflow

---

## 📊 Dataset

**Source:** CDC Diabetes Health Indicators Dataset  
**Size:** ~50,000 patient records (balanced subset)  
**Target:** `Diabetes_binary` (0 = No Diabetes, 1 = Diabetes)

### Feature Overview (21 total)
- **Demographics:** Age, Sex, Education, Income  
- **Physical measurements:** BMI, Blood Pressure  
- **Lifestyle factors:** Smoking, Physical Activity, Alcohol Consumption  
- **Medical history:** High Cholesterol, Stroke, Heart Disease  
- **Self‑reported health:** General, Mental, Physical health indicators  

🔎 *Note on class balance:*  
The original CDC dataset has ~14% diabetes prevalence. For this project, the dataset was intentionally balanced (50/50) via downsampling to simplify learning and evaluation. In production, evaluation should be performed on the natural class distribution.

---

## 🧠 Modeling Approach

- **Model type:** Multi‑Layer Perceptron (MLP)
- **Framework:** PyTorch
- **Input size:** 21 standardized features
- **Output:** Binary classification (diabetes risk)
  

### Baseline Architecture
**Design rationale:**
- ReLU activations enable learning non‑linear feature interactions
- Gradual reduction in layer size helps regularization
- Two‑output logits allow use of `CrossEntropyLoss`
- Lightweight architecture suitable for structured (non‑image) data


### Improvements Explored
- Dropout regularization
- Learning rate tuning
- Network architecture simplification
- Systematic experiment tracking and comparison

## 🔍 Project Workflow

1. Environment setup and reproducibility configuration  
2. Exploratory Data Analysis (EDA)
3. Stratified train/validation/test split (60/20/20)
4. Feature scaling and tensor conversion
5. Model design and inspection
6. Training with validation monitoring
7. Clinical‑oriented evaluation (Recall, F1, ROC‑AUC)
8. Model improvement and experimentation
9. Final healthcare interpretation and recommendations


## 📈 Key Results (Test Set)

| Metric      | Baseline | Best Model (Dropout) |
|-------------|----------|----------------------|
| Accuracy    | 0.71     | 0.75                 |
| Precision   | 0.70     | 0.73                 |
| Recall      | 0.76     | **0.78**             |
| F1‑Score    | 0.73     | **0.75**             |
| ROC‑AUC     | 0.77     | **0.82**             |

✅ The improved model meets the project goal of **≥5% performance gain** on key clinical metrics.


## 🧪 Visual Results

starter-kit/screenshots/distributions_components.png 

starter-kit/screenshots/validation_loss_comparison_by_learning_rate.png

starter-kit/screenshots/visualizing_loss_curves_top_3_experiments.png

## 🧬 Healthcare Interpretation

- **Recall** is the most important metric for diabetes screening  
- False negatives are more costly than false positives in medical contexts  
- The model is suitable as a **decision‑support screening tool**, not as a diagnostic system  
- Performance is limited by the absence of lab‑based clinical features (e.g., glucose, HbA1c)


## 🚀 Technologies Used

- **PyTorch** – Neural network modeling
- **scikit‑learn** – Preprocessing and evaluation
- **Pandas & NumPy** – Data manipulation
- **Matplotlib** – Visualization
- **Jupyter Notebook** – Experimentation and documentation

## ⚠️ Limitations & Ethics

- Not validated on real‑world prevalence data
- No clinical lab variables included
- Potential bias from self‑reported survey data
- Must not be deployed without medical oversight


## 📌 Next Steps

- Threshold tuning for recall‑prioritized screening
- Early stopping + learning rate scheduling
- Calibration and evaluation on imbalanced datasets
- Integration with richer clinical features


## 📜 License
This project is released under the MIT License.
