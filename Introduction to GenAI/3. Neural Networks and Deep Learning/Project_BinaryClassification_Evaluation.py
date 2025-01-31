import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model('cat_dog_classifier_2.keras')

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

data = np.load('valid_data.npz')

file_paths = data['file_paths']
labels = data['labels']

X_train, X_temp, y_train, y_temp = train_test_split(file_paths, labels, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

def create_generator(file_paths, labels, target_size=(150, 150), batch_size=32, augment=False):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20 if augment else 0,
        width_shift_range=0.2 if augment else 0,
        height_shift_range=0.2 if augment else 0,
        shear_range=0.2 if augment else 0,
        zoom_range=0.2 if augment else 0,
        horizontal_flip=True if augment else False,
        fill_mode='nearest'
    )
    
    generator = datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': file_paths, 'class': labels}),
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return generator

test_generator = create_generator(X_test, y_test)

y_pred_prob = model.predict(test_generator) 
y_pred = (y_pred_prob > 0.5).astype(int) 
y_true = test_generator.labels

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()