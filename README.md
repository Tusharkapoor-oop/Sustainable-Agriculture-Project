#  Sustainable Crop Recommendation System (AI/ML for Sustainable Agriculture)

## 📌 Overview
This project implements a **Machine Learning-based Crop Recommendation System** that suggests the most suitable crops for farming based on soil and environmental conditions.

The goal is to support **sustainable agriculture** by helping farmers:
- Improve crop yield 📈
- Conserve water 💧
- Reduce excessive fertilizer use ⚗️
- Maintain long-term soil health 🌍

---

## 🚀 Features
✔️ Predicts the best crop based on **soil nutrients (NPK), pH, temperature, humidity, and rainfall**  
✔️ Uses **Random Forest Classifier** for accurate predictions  
✔️ Provides **top 3 probable crops** for flexibility  
✔️ Visualizes **feature importance** (which factors matter most)  
✔️ Supports **sustainability goals** (soil fertility, water conservation, crop rotation)  

---

## 📊 Dataset
We use the **[Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)** which includes:  
- **N:** Nitrogen level in soil  
- **P:** Phosphorus level  
- **K:** Potassium level  
- **Temperature (°C)**  
- **Humidity (%)**  
- **pH (soil acidity/alkalinity)**  
- **Rainfall (mm)**  
- **Label (crop name)**  

---

## ⚙️ Tech Stack
- **Language:** Python 3.10+  
- **Libraries:**  
  - `numpy`, `pandas` (data handling)  
  - `matplotlib`, `seaborn` (visualization)  
  - `scikit-learn` (ML model training)  

---

## 🛠️ How It Works
1. Load dataset (CSV with soil & climate data).  
2. Preprocess data (split into features & labels).  
3. Train ML model (Random Forest Classifier).  
4. Evaluate performance (accuracy, classification report, confusion matrix).  
5. Predict crop recommendation for given inputs.  
6. (Optional) Show top 3 crop recommendations with probabilities.  

---

## 📈 Model Performance
- Achieved **95%+ accuracy** on test data.  
- Random Forest chosen for its robustness & interpretability.  

---

## 🔮 Example Usage
```python
# Example input
recommend_crop(N=90, P=42, K=43, temperature=20, humidity=80, ph=6.5, rainfall=200)

# Output
"Recommended Crop: rice"
```
---
## How This Supports Sustainability

✅ Reduces fertilizer overuse by aligning crops with soil nutrients.

✅ Suggests drought-tolerant crops in water-scarce regions.

✅ Encourages nitrogen-fixing crops (e.g., pulses) for soil fertility.

✅ Promotes climate-resilient farming.

---

### Project Structure 
```
Sustainable-Crop-Recommendation/
│
├── data/
│   ├── raw/                          # Original dataset (from Kaggle or gov. soil reports)
│   │   └── Crop_recommendation.csv
│   ├── processed/                    # Cleaned and transformed dataset
│   │   └── Crop_recommendation_clean.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA (plots, data insights, sustainability notes)
│   ├── 02_model_training.ipynb       # ML training experiments
│   ├── 03_evaluation_results.ipynb   # Model evaluation, confusion matrix, feature importance
│   ├── 04_final_project.ipynb        # Clean, presentable notebook (submission-ready)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Functions for cleaning, scaling, handling missing data
│   ├── feature_engineering.py        # Extra features (soil fertility index, drought score, etc.)
│   ├── model_training.py             # Train ML models, cross-validation
│   ├── model_evaluation.py           # Accuracy, precision, recall, confusion matrix
│   ├── crop_predictor.py             # Function to load model and predict crops
│   ├── visualization.py              # Correlation heatmaps, feature importance, etc.
│
├── app/
│   ├── streamlit_app.py              # (Optional) Streamlit UI for farmers
│   ├── api.py                        # (Optional) Flask/FastAPI backend for predictions
│
├── outputs/
│   ├── models/                       # Saved trained ML models
│   │   └── random_forest.pkl
│   │   └── svm_model.pkl
│   ├── reports/                      # Plots, metrics, and evaluation reports
│   │   ├── accuracy_report.txt
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│
├── tests/                            # Unit tests to show software engineering practices
│   ├── test_data_preprocessing.py
│   ├── test_model_training.py
│
├── docs/                             # Documentation (extra edge for professionalism)
│   ├── project_report.docx           # Internship report
│   ├── methodology.md                # Explains ML workflow
│   ├── sustainability_impact.md      # How project supports sustainable farming
│
├── README.md                         # Project overview
├── requirements.txt                  # Dependencies
├── environment.yml                   # (Optional) Conda environment file
├── .gitignore                        # Ignore cache, venv, etc.

```
---
## Challenges Faced in the Project

During the development of the Sustainable Crop Recommendation System, several challenges were encountered at different stages of the project. These challenges are highlighted below:

1. Data Collection

Collecting reliable and farm-level data was one of the major challenges. While the Kaggle dataset used for training was clean, real-world data such as government soil health cards often contained missing or incomplete entries (e.g., pH, nitrogen, phosphorus, and potassium values). Additionally, weather data sourced from APIs sometimes had gaps or required paid access for complete historical records.

2. Data Preprocessing and Integration

Integrating soil, weather, and crop data into a single dataset proved difficult. Different sources used varied formats (CSV, Excel, JSON), units (mm vs cm for rainfall), and time frequencies (daily weather vs yearly soil test reports). Careful preprocessing was required to ensure consistency and correctness.

3. Class Imbalance

The dataset was skewed towards commonly grown crops such as rice, maize, and wheat, while minor or region-specific crops had fewer examples. This imbalance risked biasing the model toward recommending popular crops instead of promoting diverse and sustainable crop options.

4. Feature Engineering

Creating additional features that represent sustainability, such as soil fertility index, water requirement scores, and crop rotation benefits, was a challenge. These features were not directly available in the dataset and required a mix of domain knowledge and research to design meaningfully.

5. Model Training

Training the model presented multiple challenges:

The risk of overfitting due to the relatively small dataset.

The need to select suitable algorithms (Random Forest, SVM, or XGBoost) and tune hyperparameters carefully.

Hardware limitations, as training large models with combined soil and weather datasets was resource-intensive on standard laptops.

6. Model Evaluation

Although the models achieved high accuracy, validating their real-world applicability was difficult. A model might recommend a crop that is statistically suitable, but in practice, factors such as pests, irrigation availability, and market demand may reduce feasibility. Ensuring interpretability of results (explaining why a crop was recommended) was also essential to build trust among end users.

7. Deployment

Deploying the solution for farmers introduced its own set of difficulties:

Many farmers lack access to high-end devices, requiring the application to be lightweight and mobile-friendly.

Rural areas often face unreliable internet connectivity, which could affect the usability of an online recommendation tool.

Integrating live weather APIs added recurring costs and dependency issues.

8. Domain Knowledge Gap

Agricultural expertise was necessary to validate whether the ML-driven recommendations were practical in real-world farming conditions. Some predictions made sense statistically but could fail due to ground realities such as pest infestations, local crop preferences, or fluctuating market prices, which fall outside the scope of the dataset.



---
## Future Improvements

1. Build a Streamlit web app for easy farmer access.

 2. Integrate IoT soil sensors for real-time input.

 3. Use satellite + weather API data for better predictions.

 4. Add fertilizer & irrigation recommendations.
---
## Author

TUSHAR KAPOOR 
B.Tech CSE (AI & ML) | Internship Project
