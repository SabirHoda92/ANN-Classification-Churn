import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import  EarlyStopping,TensorBoard
import datetime

from ANN.app import x_train, y_train, x_test, y_test

print(x_train.shape[1])
#build our model
#its acctually become a entire model
model = Sequential([
      Dense(64,activation='relu', input_shape=(x_train.shape[1],)), #First Hidden Layer
      Dense(32, activation='relu'), #Second hidden layer (for this time we dont need to add input shape because in Sequential its connected to each layer
      Dense(1, activation='sigmoid') #its a output layer
])

# We use model.summary() to display a summary of the model architecture.
# It shows each layer, its type, output shape, and the number of trainable parameters.
# This helps us verify that the model is built as intended before training.
# print(model.summary())
# We are defining the Adam optimizer with a custom learning rate.
# Adam (Adaptive Moment Estimation) combines the benefits of AdaGrad and RMSProp,
# making it efficient and widely used for training deep learning models.
# The learning_rate=0.01 controls the step size for weight updates:
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.BinaryCrossentropy()

#compile the model
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

log_dir= "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# We use TensorBoard callback to visualize and monitor training in real-time.
# It helps track metrics like loss, accuracy, learning rate, and model graphs.
# After training, we can open TensorBoard in the browser to see plots and comparisons.
# Example usage: tensorboard --logdir=log_dir
tensorflow_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#setup Early Stopping
# We use EarlyStopping to stop training automatically when the model stops improving.
# monitor='val_loss' → watches the validation loss after each epoch.
# patience=5 → waits for 5 epochs without improvement before stopping.
# restore_best_weights=True → after stopping, restores the model weights from the epoch
# where the validation loss was best (prevents overfitting).
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#Train the model
history= model.fit(
      x_train,y_train,validation_data=(x_test,y_test), epochs= 100,
      callbacks=[tensorflow_callback,early_stopping_callback]
)

#save the model
# We use model.save('model.h5') to save the entire model in HDF5 (.h5) format.
# This format stores:
#   - Model architecture (layers and connections)
#   - Weights of the model
#   - Training configuration (loss, optimizer, metrics)
model.save('model.h5')

#load tensorboard Extension
%load_ext tensorboard



