# Customer Churn Prediction - ANN Classification Project

A deep learning project using Artificial Neural Networks (ANN) to predict customer churn. This project includes data preprocessing, model training, and both Jupyter notebook and Streamlit web interface for making predictions.

## Project Overview

This project builds a binary classification model to predict whether a customer is likely to churn (leave the bank). The model is trained on customer data including demographics, account information, and activity metrics.

**Key Features:**
- Artificial Neural Network with Keras/TensorFlow
- Data preprocessing with encoding and scaling
- Early stopping to prevent overfitting
- TensorBoard integration for training visualization
- Streamlit web interface for interactive predictions
- Jupyter notebooks for experimentation and analysis

## Dataset

- **File:** `Churn_Modelling.csv`
- **Features:** 13 input features including:
  - Geography (France, Germany, Spain)
  - Gender
  - Credit Score
  - Age, Tenure
  - Balance, Number of Products
  - Credit Card status, Active Member status
  - Estimated Salary
- **Target:** Churn (Binary: 0 = No, 1 = Yes)

## Project Structure

```
ANN_classification_project/
├── app.py                          # Streamlit web application
├── exeriments.ipynb               # Model training notebook
├── prediction.ipynb               # Prediction examples notebook
├── hyperparameter.ipynb           # Hyperparameter tuning notebook
├── Churn_Modelling.csv            # Dataset
├── model.h5                       # Trained model
├── requirements.txt               # Python dependencies
├── label_encoder.pkl              # Gender encoder
├── One_hot_encoder.pkl            # Geography encoder
├── Scaler.pkl                     # Feature scaler
├── logs/                          # TensorBoard logs
└── env/                           # Virtual environment
```

## Installation & Setup

### 1. Create Virtual Environment

```bash
python -m venv env
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
& .\env\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
env\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- TensorFlow/Keras
- scikit-learn
- pandas, numpy
- Streamlit
- matplotlib, seaborn

## Usage

### Option 1: Streamlit Web App (Recommended)

Run the interactive web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Select customer geography
- Input customer details using form controls
- Get instant churn prediction with probability
- User-friendly interface

### Option 2: Jupyter Notebooks

#### Training Notebook (`exeriments.ipynb`)
- Data loading and exploration
- Data preprocessing (encoding, scaling)
- Model architecture definition
- Model training with callbacks
- Model evaluation

#### Prediction Notebook (`prediction.ipynb`)
- Load trained model and encoders
- Example prediction workflow
- Manual feature engineering for new customers

Run notebooks with:
```bash
jupyter notebook
```

## Model Architecture

**Neural Network Structure:**
- **Input Layer:** 12 features (after one-hot encoding)
- **Hidden Layer 1:** 64 neurons with ReLU activation
- **Hidden Layer 2:** 32 neurons with ReLU activation
- **Output Layer:** 1 neuron with Sigmoid activation (binary classification)

**Training Configuration:**
- Optimizer: Adam (learning_rate=0.01)
- Loss: Binary Crossentropy
- Metrics: Accuracy
- Early Stopping: Patience=10 epochs
- TensorBoard logging enabled

## Preprocessing Pipeline

### 1. Encoding
- **Geography:** One-Hot Encoding (France, Germany, Spain)
- **Gender:** Label Encoding (Male=1, Female=0)

### 2. Scaling
- StandardScaler for all numeric features
- Fitted on training data, applied to test/prediction data

### 3. Feature Order
Critical for predictions - must match training order:
```
['Geography_France', 'Geography_Germany', 'Geography_Spain', 
 'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
```

## Making Predictions

### Using Streamlit App
1. Open the app in browser
2. Fill in customer information
3. Click predict
4. View churn probability and prediction

### Programmatically

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

# Load model and encoders
model = load_model('model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('One_hot_encoder.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [600],
    'Gender': [label_encoder_gender.transform(['Male'])[0]],
    'Age': [40],
    'Tenure': [3],
    'Balance': [60000],
    'NumOfProducts': [2],
    'HasCrCard': [1],
    'IsActiveMember': [1],
    'EstimatedSalary': [50000]
})

# Encode Geography
geo_encoded = onehot_encoder_geo.transform([['France']]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine and reorder
input_data = pd.concat([geo_df, input_data.reset_index(drop=True)], axis=1)
column_order = ['Geography_France', 'Geography_Germany', 'Geography_Spain', 
                'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
                'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
input_data = input_data[column_order]

# Scale and predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
churn_prob = prediction[0][0]

print(f"Churn Probability: {churn_prob:.2%}")
print(f"Prediction: {'Likely to churn' if churn_prob > 0.5 else 'Not likely to churn'}")
```

## Troubleshooting

### Feature Name Error
**Error:** `ValueError: The feature names should match those that were passed during fit.`

**Solution:** Ensure columns are reordered to match training order before scaling.

### Model Not Found
**Error:** `FileNotFoundError: model.h5`

**Solution:** Make sure you're running commands from the project directory and model.h5 is present.

### Missing Encoders
**Error:** `FileNotFoundError: One_hot_encoder.pkl` or similar

**Solution:** Run the training notebook (exeriments.ipynb) first to generate all pickle files.

## Training Visualization

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/fit
```

Then open `http://localhost:6006` in your browser to see:
- Training/validation loss
- Training/validation accuracy
- Histograms of weights and biases

## Performance Metrics

The model is evaluated on a test set (20% of data) with:
- Accuracy
- Loss (Binary Crossentropy)
- Early stopping based on validation loss

## Model Comparison: Random Forest vs ANN

To optimize performance and ensure the best model for production, we compared **Random Forest** (traditional ML) with **ANN** (deep learning) on the same dataset.

### Overall Performance Summary

| Metric | Random Forest | ANN | Winner |
|--------|---------------|-----|--------|
| **Accuracy** | 86.70% | 86.55% | 🏆 RF |
| **Precision (Churn)** | 75.50% | 77.20% | 🏆 ANN |
| **Recall (Churn)** | 47.80% | 44.80% | 🏆 RF |
| **F1-Score (Churn)** | 58.20% | 57.60% | 🏆 RF |
| **Specificity** | 96.20% | 96.80% | 🏆 ANN |
| **ROC-AUC** | High | High | Comparable |

### Confusion Matrix Analysis

#### Random Forest
```
                True Negative  False Positive
No Churn:           1546             61
Churn:               205            188
```
- **True Positives (Churners Caught):** 188/393 (47.8%)
- **False Negatives (Churners Missed):** 205/393 (52.2%)

#### ANN
```
                True Negative  False Positive
No Churn:           1555             52
Churn:               217            176
```
- **True Positives (Churners Caught):** 176/393 (44.8%)
- **False Negatives (Churners Missed):** 217/393 (55.2%)

### Key Insights

#### 1. **Churn Detection (Most Important for Business)**
- **Random Forest catches 12 more churners** than ANN (188 vs 176)
- This translates to 3% higher recall on the churn class
- **Business Impact:** Better at identifying at-risk customers for retention campaigns

#### 2. **False Positive Rate**
- **Random Forest:** 61 false positives (wastes retention resources on 61 non-churners)
- **ANN:** 52 false positives (slightly more efficient resource allocation)
- **Trade-off:** RF catches more churners but has 9 more false alarms

#### 3. **Model Characteristics**

**Random Forest 🌲**
- ✅ Catches more actual churners (higher recall)
- ✅ Better F1-score (balanced precision-recall)
- ✅ Faster predictions
- ✅ Easier to interpret (feature importance available)
- ✅ Smaller model (easier to deploy)
- ⚠️ Slightly lower precision

**ANN 🧠**
- ✅ Higher precision (fewer false alarms)
- ✅ Higher specificity (better at identifying non-churners)
- ✅ Flexible architecture for future improvements
- ⚠️ Catches fewer churners (lower recall)
- ⚠️ Black-box model (harder to explain)
- ⚠️ Requires TensorFlow/GPU (complex infrastructure)

### Business Recommendation

**🎯 DEPLOY: Random Forest**

#### Rationale:
1. **Higher Churn Detection Rate** - Catches 47.8% of actual churners vs ANN's 44.8%
   - 12 additional churners identified = more customers saved
   
2. **Better Overall Accuracy** - 86.70% vs 86.55% on test set

3. **Superior F1-Score** - 58.2% vs 57.6% (balanced performance)

4. **Operational Benefits:**
   - ✅ Simpler to deploy and maintain
   - ✅ No deep learning framework dependencies
   - ✅ Fast inference time (real-time predictions)
   - ✅ Interpretable model (understand which features drive churn)
   - ✅ Smaller model size (version control friendly)

5. **Production Readiness:**
   - Easier to retrain and update
   - No GPU required
   - More stable in production environment
   - Better for explaining predictions to business stakeholders

### Performance Visualization

See [`model_comarision.ipynb`](model_comarision.ipynb) for detailed visualizations including:
- ROC-AUC curves comparison
- Confusion matrices side-by-side
- Metric-by-metric bar charts
- Classification reports

### Conclusion

While both models perform similarly (86.7% vs 86.55% accuracy), **Random Forest is the optimal choice** for this customer churn prediction problem. It provides the best balance between catching actual churners, operational simplicity, and production-readiness.

## Requirements

See `requirements.txt` for complete list. Key dependencies:
- TensorFlow >= 2.x
- scikit-learn
- pandas
- numpy
- streamlit
- matplotlib
- seaborn

## Notes

- The model uses early stopping to prevent overfitting
- Feature scaling is critical for model performance
- One-hot encoding order matters for Geography (France, Germany, Spain)
- Always maintain consistent preprocessing between training and prediction

## Future Improvements

- Hyperparameter tuning (layer sizes, learning rates, dropout)
- Feature engineering (interaction terms, polynomial features)
- Class imbalance handling (SMOTE, class weights)
- Model interpretability (SHAP values, feature importance)
- Cross-validation for better performance estimates
- API deployment (FastAPI, Flask)

## Author

Data Science Project - Customer Churn Prediction

## License

Open source for educational purposes
