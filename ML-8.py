# Rewriting the full code to execute and show the confusion matrix graph

# Step 1: Import the required modules
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 2: Generate the dataset
x, y = make_classification(
    n_samples=100,
    n_features=1,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    n_informative=1,
    n_redundant=0,
    n_repeated=0,
    random_state=42
)

# Step 3: Visualize the data
plt.figure(figsize=(6, 4))
plt.scatter(x, y, c=y, cmap='rainbow')
plt.title('Scatter Plot of Logistic Regression')
plt.xlabel('Feature')
plt.ylabel('Class')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# Step 5: Perform Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

# Step 6: Make prediction using the model
y_pred = log_reg.predict(x_test)

# Step 7: Display and plot the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_reg.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
