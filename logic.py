import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

# Load the preprocessed dataset
preprocessed_file_path = 'preprocessed_flowdata4.csv'
data = pd.read_csv(preprocessed_file_path)

# Extract features and target variable
X = data.drop(columns=['label'])
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a function to evaluate and save model results
def evaluate_and_save_model(model, model_name, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
   
    # Make predictions
    y_pred = model.predict(X_test)
   
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
   
    # Specify the path for the results file
    results_file_path = f'{model_name}_results.txt'
   
    # Write results to the file
    with open(results_file_path, 'w') as file:
        file.write(f'Accuracy: {accuracy}\n')
        file.write('Classification Report:\n')
        file.write(report)
   
    print(f'{model_name} results have been written to {results_file_path}')

# Initialize and evaluate Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
evaluate_and_save_model(logistic_model, 'logistic_regression', X_train, X_test, y_train, y_test)

# Initialize and evaluate K-Nearest Neighbors
knn_model = KNeighborsClassifier()
evaluate_and_save_model(knn_model, 'knn', X_train, X_test, y_train, y_test)

# Initialize and evaluate Decision Tree
dt_model = DecisionTreeClassifier()
evaluate_and_save_model(dt_model, 'decision_tree', X_train, X_test, y_train, y_test)

# Initialize and evaluate Gaussian Naive Bayes
gnb_model = GaussianNB()
evaluate_and_save_model(gnb_model, 'gaussian_naive_bayes', X_train, X_test, y_train, y_test)

# Initialize and evaluate Random Forest
rf_model = RandomForestClassifier()
evaluate_and_save_model(rf_model, 'random_forest', X_train, X_test, y_train, y_test)




# Path to the new data file
new_data_file_path = 'preprocessed_flowdata11.csv'

# Check if the file exists
if os.path.exists(new_data_file_path):
    new_data = pd.read_csv(new_data_file_path)
    
    # Extract features from the new data (no target variable)
    X_new = new_data.drop(columns=['label'], errors='ignore')  # Assuming 'label' might not be present

    # Standardize the new data using the same scaler
    X_new = scaler.transform(X_new)

    # Predict using the KNN model
    rf_predictions = rf_model.predict(X_new)

    # Print the predictions
    print('Predictions for the new data:')
    print(rf_predictions)

    # Optionally, save the predictions to a file
    predictions_file_path = 'rf_predictions.csv'
    pd.DataFrame(rf_predictions, columns=['Predicted_Label']).to_csv(predictions_file_path, index=False)

    print(f'Predictions have been written to {predictions_file_path}')
else:
    print(f'Error: File {new_data_file_path} not found.')