# Loan Prediction with TensorFlow and Keras

## Overview
This project focuses on predicting loan approval using machine learning techniques with TensorFlow and Keras. The dataset used in this project is derived from LendingClub loan data, containing features such as loan amount, interest rates, employment details, home ownership status, and credit history.

## Features
- **Data Preprocessing**: Handling missing values, categorical encoding, and feature scaling.
- **Exploratory Data Analysis (EDA)**: Statistical summaries, visualizations, and correlation analysis.
- **Model Building**: Implementing a deep learning model with TensorFlow and Keras.
- **Evaluation Metrics**: Assessing model performance using accuracy, precision, recall, and F1-score.
- **Hyperparameter Tuning**: Optimizing model performance using tuning techniques.

## Project Structure
```
loan-prediction-with-tensor-flow-and-keras/
│-- data/                     # Raw dataset and processed data
│-- notebooks/                # Jupyter Notebooks for analysis and model training
│-- src/                      # Python scripts for preprocessing and model training
│-- models/                   # Saved trained models
│-- results/                  # Evaluation results and model performance metrics
│-- README.md                 # Project documentation
│-- requirements.txt          # Python dependencies
│-- config.yaml               # Configuration settings
│-- app.py                    # Web application for loan prediction (if applicable)
```

## Dataset
The dataset contains financial and demographic information about borrowers. The key features include:
- `loan_amnt`: Loan amount requested by the borrower.
- `term`: Loan repayment period (e.g., 36 or 60 months).
- `int_rate`: Interest rate on the loan.
- `emp_length`: Length of employment in years.
- `home_ownership`: Type of home ownership (e.g., Rent, Own, Mortgage).
- `annual_inc`: Annual income of the borrower.
- `dti`: Debt-to-income ratio.
- `revol_util`: Credit utilization rate.
- `loan_status`: Target variable (Fully Paid or Charged Off).

## Installation
To set up and run the project, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/loan-prediction.git
   cd loan-prediction
   ```

2. **Create a virtual environment** (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Run Jupyter Notebook**:
   ```sh
   jupyter notebook
   ```

## Model Training
- The model is built using a sequential neural network with fully connected dense layers.
- Activation functions such as ReLU and Sigmoid are used.
- The loss function is binary cross-entropy as the problem is a classification task.
- Adam optimizer is used for gradient descent optimization.
- The model is trained on processed data with appropriate train-test splits.

## Evaluation Metrics
- **Accuracy**: Measures overall correctness of predictions.
- **Precision & Recall**: Useful for assessing false positives and false negatives.
- **ROC-AUC Score**: Evaluates the classification model's performance.
- **Confusion Matrix**: Visual representation of classification results.

## Future Improvements
- Implement additional feature engineering techniques.
- Experiment with different neural network architectures.
- Apply hyperparameter tuning using GridSearchCV or Bayesian Optimization.
- Deploy the model using Flask or FastAPI for real-time predictions.

## Contributions
Feel free to contribute to this project by submitting issues or pull requests.
