import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

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
    report = classification_report(y_test, y_pred)
   
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

# Initialize and evaluate Random Forest
rf_model = RandomForestClassifier()
evaluate_and_save_model(rf_model, 'random_forest', X_train, X_test, y_train, y_test)

# Initialize and evaluate Support Vector Machine
svm_model = SVC()
evaluate_and_save_model(svm_model, 'svm', X_train, X_test, y_train, y_test)

# Initialize and evaluate Neural Network (MLPClassifier)
mlp_model = MLPClassifier(max_iter=1000)
evaluate_and_save_model(mlp_model, 'neural_network', X_train, X_test, y_train, y_test)

# Initialize and evaluate Gradient Boosting Classifier
gbc_model = GradientBoostingClassifier()
evaluate_and_save_model(gbc_model, 'gradient_boosting', X_train, X_test, y_train, y_test)

# Initialize and evaluate AdaBoost Classifier
ada_model = AdaBoostClassifier()
evaluate_and_save_model(ada_model, 'adaboost', X_train, X_test, y_train, y_test)

# Initialize and evaluate XGBoost Classifier
xgb_model = xgb.XGBClassifier()
evaluate_and_save_model(xgb_model, 'xgboost', X_train, X_test, y_train, y_test)

# Initialize and evaluate LightGBM Classifier
lgb_model = lgb.LGBMClassifier()
evaluate_and_save_model(lgb_model, 'lightgbm', X_train, X_test, y_train, y_test)

# Initialize and evaluate CatBoost Classifier
catboost_model = CatBoostClassifier(verbose=0)
evaluate_and_save_model(catboost_model, 'catboost', X_train, X_test, y_train, y_test)

# Initialize and evaluate Extra Trees Classifier
etc_model = ExtraTreesClassifier()
evaluate_and_save_model(etc_model, 'extra_trees', X_train, X_test, y_train, y_test)

# Optionally, save predictions for one of the models (e.g., KNN)
# Load and preprocess the new data
new_data_file_path = 'new_data.csv'
new_data = pd.read_csv(new_data_file_path)

# Extract features from the new data (no target variable)
X_new = new_data.drop(columns=['label'], errors='ignore')  # Assuming 'label' might not be present

# Standardize the new data using the same scaler
X_new = scaler.transform(X_new)

# Predict using the KNN model
knn_predictions = knn_model.predict(X_new)

# Print the predictions
print('Predictions for the new data:')
print(knn_predictions)

# Optionally, save the predictions to a file
predictions_file_path = 'knn_predictions.csv'
pd.DataFrame(knn_predictions, columns=['Predicted_Label']).to_csv(predictions_file_path, index=False)

print(f'Predictions have been written to {predictions_file_path}')
