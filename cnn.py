import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load datasets with proper paths
train_data = pd.read_csv(r'C:\Users\Arghya\sign language\archive\sign_mnist_train.csv')
test_data = pd.read_csv(r'C:\Users\Arghya\sign language\archive\sign_mnist_test.csv')

# Separate features and labels
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Check unique values in the labels to confirm label ranges
unique_y_train = np.unique(y_train)
unique_y_test = np.unique(y_test)
print("Unique values in y_train:", unique_y_train)
print("Unique values in y_test:", unique_y_test)

# Normalize pixel values between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape data to match the input shape of CNN (28x28 grayscale images)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Number of classes in the dataset
num_classes = 24  # Letters A-Z excluding J

# Adjust labels: If 'J' is skipped, remap labels after 'J'
y_train[y_train > 9] -= 1  # Labels > 9 are shifted down by 1
y_test[y_test > 9] -= 1

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)

# Save the model
model.save("sign_language_model.h5")

# Optionally, print out the training history for future inspection
print("Training history:", history.history)
