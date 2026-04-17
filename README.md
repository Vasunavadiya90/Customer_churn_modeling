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
