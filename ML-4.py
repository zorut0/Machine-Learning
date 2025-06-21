# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load the Iris dataset from CSV
iris = pd.read_csv("iris_500.csv")

# Step 2: Split features and target
X = iris.iloc[:, :-1]             # Features (all columns except last)
y = iris.iloc[:, -1]              # Target (last column: species names)
target_names = y.unique()         # Get unique class labels

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Create a DataFrame with the reduced data
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['target'] = y

# Step 6: Plot the PCA result
plt.figure(figsize=(8,6))
colors = ['red', 'green', 'blue']
for i, target_name in enumerate(target_names):
    plt.scatter(df_pca[df_pca['target'] == target_name]['PC1'],
                df_pca[df_pca['target'] == target_name]['PC2'],
                label=target_name,
                color=colors[i])
    
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of IRIS Dataset (from CSV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Explained variance ratio
print("Explained variance ratio by each principal component:")
print(pca.explained_variance_ratio_)
