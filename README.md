# Heart-Disease-Risk-Prediction




# Heart Disease Risk Prediction System

This project is an interactive machine learning web application built with **Streamlit** that predicts the risk of heart disease based on user inputs. The model is powered by an **XGBoost classifier** and offers **explainable AI insights** using **SHAP values**, making the predictions transparent and easy to interpret.



## Features

- Predicts **heart disease risk** based on 13 medical features.
- Displays **risk level** categorized from *Very Low* to *Very High*.
- Built using a high-performing **XGBoost model** with ~99% accuracy.
- Includes **SHAP visualizations** for feature importance and model explainability.
- Clean, interactive **Streamlit UI** for user-friendly experience.
- Real-time model inference and interpretation.


## Tech Stack

- **Python**
- **Scikit-learn**
- **XGBoost**
- **Streamlit**
- **Pandas, Numpy, Matplotlib**
- **SHAP (SHapley Additive exPlanations)**



##  Dataset

- The dataset is sourced from the **UCI Heart Disease Dataset**.
- It includes features like age, cholesterol level, blood pressure, etc.
- [UCI Dataset Link](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)





##  Input Features

| Feature      | Description |
|--------------|-------------|
| age          | Age in years |
| sex          | Gender (1 = male, 0 = female) |
| cp           | Chest pain type (0–3) |
| trestbps     | Resting blood pressure |
| chol         | Serum cholesterol |
| fbs          | Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) |
| restecg      | Resting electrocardiographic results (0–2) |
| thalach      | Maximum heart rate achieved |
| exang        | Exercise-induced angina (1 = yes, 0 = no) |
| oldpeak      | ST depression induced by exercise |
| slope        | Slope of peak exercise ST segment (0–2) |
| ca           | Number of major vessels (0–3) |
| thal         | Thalassemia (0–3) |



##  Risk Level Interpretation

| Probability | Risk Category |
|-------------|----------------|
| 1–20%       | 🟢 Very Low |
| 21–40%      | 🟢 Low |
| 41–60%      | 🟡 Medium |
| 61–80%      | 🟠 High |
| 81–99%      | 🔴 Very High |



##  Acknowledgements

- UCI Machine Learning Repository
- Streamlit Team
- XGBoost & SHAP developers





## Contact

Made by Aman Sharma  
Amytheachiever6263@gmail.com • Linked in  - https://www.linkedin.com/in/aman-sharma-9a9592255/
