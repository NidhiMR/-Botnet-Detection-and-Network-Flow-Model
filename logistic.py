import pandas as pd

# Load the dataset
file_path = 'flowdata11.binetflow.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Display the summary of the dataset
print(data.info())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Handle missing values
data['state'].fillna('UNKNOWN', inplace=True)
data['stos'].fillna(data['stos'].mean(), inplace=True)
data['dtos'].fillna(data['dtos'].mean(), inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
data['proto'] = label_encoder.fit_transform(data['proto'])
data['dir'] = label_encoder.fit_transform(data['dir'])
data['state'] = label_encoder.fit_transform(data['state'])

# Extract features and target variable
X = data.drop(columns=['label', 'Family'])
y = data['label']

# Encode the target variable
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Combine the features and target variable for the preprocessed data
preprocessed_data = pd.DataFrame(X, columns=X.columns)
preprocessed_data['label'] = y

# Save the preprocessed data to a new CSV file
preprocessed_file_path = 'preprocessed_flowdata11.csv'
preprocessed_data.to_csv(preprocessed_file_path, index=False)

print(f'Preprocessed data saved to {preprocessed_file_path}')



