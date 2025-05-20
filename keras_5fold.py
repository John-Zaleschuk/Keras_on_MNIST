#============================================#
# Author:  John Zaleschuk                    #
# Purpose: Using keras on the MNIST data set #
#============================================#
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

NUM_EPOCHS = 8
# model training output
# 0 for none, 1 for all, 2 for some.
VERBOSE_VAL = 0

# Read data and separate labels and features
raw_data = pd.read_csv("MNIST.csv")
X = raw_data.iloc[:, 1:].values
Y = raw_data.iloc[:, 0].values

# Normalize pixel values to [0, 1]
X = X / 255.0

# Initialize KFold with default 5 splits, random state for repeatability
kf = KFold(shuffle=True, random_state=8)
test_acc_array = []
train_acc_array = []

# Train and evaluate the model on each fold
for i, (train_index, test_index) in enumerate(kf.split(X)):
    train_X = X[train_index]
    train_Y = Y[train_index]
    test_X = X[test_index]
    test_Y = Y[test_index]

    # Model Definition
    # uses keras sequential fully connected (dense) layers
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(784,)),
        # two hidden layers using relu activation
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        # final layer is 10, for 0-9 digits
        tf.keras.layers.Dense(10)
    ])

    # Compile the model, from_logits=True will use softmax activation
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model, capturing scores
    scores = model.fit(train_X, train_Y, epochs=NUM_EPOCHS, validation_data=(test_X, test_Y), verbose=VERBOSE_VAL)

    # Evaluate model on the test set
    test_loss, test_accuracy = model.evaluate(test_X, test_Y, verbose=VERBOSE_VAL)
    test_acc_array.append(test_accuracy)
    train_acc_array.append(scores.history['accuracy'])

    print(f" Fold {i+1} Test Accuracy: {test_accuracy:.4f}")

# Calc average test accuracy across all folds
avg_accuracy = np.mean(test_acc_array)
print(f"\nAverage Test Accuracy: {avg_accuracy:.4f}")

# Plot the learning curves for each fold
plt.figure(figsize=(8, 6))
for i, i_acc in enumerate(train_acc_array):
    plt.plot(range(1, (NUM_EPOCHS+1)), i_acc, label=f"Fold {i+1}")
plt.xlabel('Epochs')
plt.ylabel('Train Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()