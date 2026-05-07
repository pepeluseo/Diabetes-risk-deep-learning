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
``
