import os
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

data = np.load('valid_data.npz')

file_paths = data['file_paths']
labels = data['labels']

X_train, X_temp, y_train, y_temp = train_test_split(file_paths, labels, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

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

train_generator = create_generator(X_train, y_train, augment=True)
validation_generator = create_generator(X_val, y_val)
test_generator = create_generator(X_test, y_test)

def dataframe_generator():
    for batch in train_generator:
        yield batch[0], batch[1] 

train_dataset = tf.data.Dataset.from_generator(
    dataframe_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 150, 150, 3), dtype=tf.float32),  # Input shape
        tf.TensorSpec(shape=(None,), dtype=tf.float32)               # Label shape
    )
).repeat()

model = models.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Fourth convolutional block
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Fifth convolutional block
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten the output
    layers.Flatten(),

    # Fully connected layers
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5, seed=seed), 
    layers.Dense(1, activation='sigmoid') 
])

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy',
    metrics=['accuracy'] 
)

model.summary()

history = model.fit(
    train_dataset,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20,
    shuffle=False
)

model.save('cat_dog_classifier_2.keras')