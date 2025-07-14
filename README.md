# General_health_prediction-
A machine learning project to predict general health status using the cleaned CVD dataset with SVM, Random Forest, and Logistic Regression, including accuracy comparison and hyperparameter tuning.

#  General Health Prediction using Machine Learning

This project predicts the **general health status** of individuals using cardiovascular-related features from a cleaned dataset. It demonstrates the application of machine learning models in health analytics and simulates an AI-based health prediction system.

##  Dataset

The dataset `CVD_cleaned.csv` contains various features like:
- Age, Gender, Blood Pressure, Cholesterol, BMI, etc.
- Target variable: General Health Status (multi-class)

##  Project Workflow

1. **Data Loading & Preprocessing**  
   - Cleaned and structured data used for training ML models.

2. **Train-Test Split**  
   - 80:20 ratio for training and testing datasets.

3. **Model Building**  
   - Trained three classifiers:
     - Support Vector Machine (SVM)
     - Random Forest Classifier
     - Logistic Regression

4. **Model Evaluation**  
   - Accuracy Score
   - Classification Report
   - Confusion Matrix

5. **Visualization**  
   - Bar chart comparing model accuracies.

6. **Hyperparameter Tuning**  
   - Used `RandomizedSearchCV` to improve model performance.

7. **AI Simulation**  
   - Simulated a single input prediction to mimic real-time health status prediction by an AI system.

##  Tools & Libraries Used

- Python (Jupyter Notebook)
- Pandas, NumPy
- Scikit-learn (SVM, RF, LR, metrics, model_selection)
- Matplotlib, Seaborn

##  Output Highlights

- Clear comparison of model performance
- Real-time prediction output
- Best model after tuning

##  How to Run

1. Open `General_health_prediction.ipynb` in Jupyter Notebook or Google Colab.
2. Upload `CVD_cleaned.csv` when prompted.
3. Run all cells sequentially to:
   - Train and evaluate models
   - Visualize performance
   - Predict single sample



