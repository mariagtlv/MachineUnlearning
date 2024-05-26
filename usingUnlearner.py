import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, losses
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

# Assuming unlearner is implemented for TensorFlow (unlearner_tf)
from unlearner.CNNUnlearner import CNNUnlearnerMedium

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
print('Datasets are built')

# Normalize images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0



# Add a channels dimension (needed for convolutional layers in TensorFlow)
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Change labels from 6 to 3
train_labels[train_labels == 6] = 3
test_labels[test_labels == 6] = 3
print('Now all 6s are 3s')

# Split training data into training and validation sets
train_indices, valid_indices = train_test_split(np.arange(len(train_images)), test_size=0.1, random_state=42)
train_data = train_images[train_indices], train_labels[train_indices]
valid_data = train_images[valid_indices], train_labels[valid_indices]

test_data = test_images, test_labels

# Instantiate the CNNUnlearner (you can choose the appropriate subclass)
unlearner = CNNUnlearnerMedium(train=train_data, test=test_data, valid=valid_data)

# Call the explain_prediction method to calculate pixel influences
# Specify parameters like deletion size, batch size, etc.
influences, diverged = unlearner.explain_prediction(
    x=test_images, 
    y=test_labels, 
    deletion_size=1, 
    batch_size=500, 
    rounds=1, 
    scale=75000, 
    damping=1e-2, 
    verbose=True
)

# Analyze the results (influences variable)
# Visualize the pixel influences to understand the impact of changing pixels on the model's predictions
