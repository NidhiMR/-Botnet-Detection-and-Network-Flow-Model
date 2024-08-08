import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'flowdata4.binetflow.csv'  # Replace with the actual path to your dataset
data = pd.read_csv(file_path)

# Filter the data for the botnet label
botnet_data = data[data['label'].str.contains('botnet', case=False, na=False)]

# Compute the correlation matrix for numerical features
corr_matrix = botnet_data[['tot_bytes', 'tot_pkts', 'src_bytes', 'dst_bytes']].corr()

# Plot heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Botnet Traffic Features')
plt.show()
