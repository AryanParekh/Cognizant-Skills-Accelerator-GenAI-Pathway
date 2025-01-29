# Weather Prediction Project with Decision Trees
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Part 1.1: Choose and Load the dataset
df = pd.read_csv('weatherHistory.csv')[:600]
df = df.drop(columns=['Precip Type','Formatted Date','Apparent Temperature (C)','Wind Bearing (degrees)', 'Visibility (km)', 'Daily Summary','Loud Cover'])
print(df.columns)

# Part 1.2: Preprocess the data
df.fillna(df.mean(numeric_only=True), inplace=True) #Handle missing values

label_encoder = LabelEncoder()
df['Summary'] = label_encoder.fit_transform(df['Summary']) #Encode the target variable summary

X = df.drop(columns=['Summary'])
y = df['Summary'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43) #Split the data into training and testing set

# Part 1.3: Train the decision tree
model = DecisionTreeClassifier(max_depth=10, min_samples_split=100, random_state=42)
model.fit(X_train, y_train)

# Part 2.1 and 3.1: Test for overfitting or underfitting, predictions on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}\n")

# Part 3.2: Evaluation
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_ if 'label_encoder' in locals() else None, yticklabels=label_encoder.classes_ if 'label_encoder' in locals() else None)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()