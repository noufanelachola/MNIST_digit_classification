import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

plt.imshow(X_train[1], cmap="gray")  
plt.title(f"Label: {y_train[1]}")  
plt.show()

print(f"Train shape : {X_train.shape}")
print(f"Test shape : {X_test.shape}")