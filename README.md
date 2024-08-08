# Botnet Detection and Network Flow Model

## Overview

The **Botnet Detection and Network Flow Model** project aims to identify network threats by analyzing network flow data. This project uses machine learning techniques to classify network traffic into various categories to detect potential botnet activities or background activities.

## Files

### 1. `logistic.py`

This script preprocesses the raw network flow data and prepares it for machine learning. The steps include:

- **Loading the dataset**: Reads the network flow data from a CSV file (`flowdata4.binetflow.csv`).[Training Dataset]
- **Data Cleaning and Encoding**:
  - Fills missing values with appropriate defaults.
  - Encodes categorical variables using `LabelEncoder`.
  - Splits the dataset into training and testing sets.
  - Standardizes the features using `StandardScaler`.
    
- **Saving Preprocessed Data**: Saves the preprocessed data to a new CSV file (`preprocessed_flowdata4.csv`).
- **Perform the same step for testing dataset (`flowdata11.binetflow.csv`)**

### 2. `login.py`

This script evaluates multiple machine learning models on the preprocessed data and makes predictions on new data. The steps include:

- **Loading the Preprocessed Dataset**: Reads the preprocessed data from a CSV file (`preprocessed_flowdata4.csv`).
- **Training and Evaluating Models**:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Gaussian Naive Bayes
  - Random Forest
  - Support Vector Classifier (SVC)
  - Neural Network (MLPClassifier)
- **Evaluation**: Each model is trained, evaluated, and results are saved to respective text files.
- **Predictions on New Data**:
  - Loads a new preprocessed data file (`preprocessed_flowdata11.csv`).
  - Uses the Random Forest model to make predictions on this new data.
  - Saves the predictions to a CSV file (`rf_predictions.csv`).

## Dataset

The dataset used in this project contains the following columns:

- **`dur`**: Duration of the network flow
- **`proto`**: Protocol used (e.g., TCP, UDP)
- **`dir`**: Direction of traffic (e.g., incoming, outgoing)
- **`state`**: Connection state (e.g., established, closed)
- **`stos`**: Source-to-destination bytes
- **`dtos`**: Destination-to-source bytes
- **`tot_pkts`**: Total number of packets
- **`tot_bytes`**: Total number of bytes
- **`src_bytes`**: Source bytes
- **`label`**: Classification label indicating the type of network flow
- **`Family`**: Classification family

### 2. `Mappings`
    -Contains a list of what each label indicates.
## How to Run

1. **Preprocess Data**:
   - Run `logistic.py` to preprocess the raw data and save it to `preprocessed_flowdata4.csv`.This is training dataset.
   - Run `logistic.py` to preprocess the raw data and save it to `preprocessed_flowdata11.csv`.This is testing dataset.


2. **Evaluate Models**:
   - Run `login.py` to train and evaluate different machine learning models on the preprocessed data. It will also make predictions on new data if available.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn

You can install the required Python packages using pip:
```bash
pip install pandas scikit-learn
