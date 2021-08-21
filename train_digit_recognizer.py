import keras
from keras.datasets import mnist 
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf 

# =========================== Dataset Development ==============================
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_mod=x_train/255.0
x_test_mod = x_test/255.0

x_train_mod = x_train_mod.reshape(len(x_train_mod),28,28,1)
x_test_mod = x_test_mod.reshape(len(x_test_mod),28,28,1)

# ========================== Model development ================================= 

def build_model(hp):
    model = keras.Sequential([
    keras.layers.Conv2D(
        filters=hp.Int('conv_1', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        input_shape=(28,28,1)), 

    layers.Dropout(0.2),
    keras.layers.Conv2D(
        filters=hp.Int('conv_2', min_value=128, max_value=256, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    ), 
    layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=256, max_value=512, step=16),
        activation='relu'
    ),
    keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 
    return model  

import kerastuner as kt
tuner = kt.Hyperband(build_model, 
                     objective='val_accuracy',
max_epochs=5,
factor=3, 
directory='dir', 
project_name='khyperband')  


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x_train_mod, y_train, epochs=5, validation_split=0.2, callbacks=[stop_early]) 

best_hp=tuner.get_best_hyperparameters()[0]

h_model = tuner.hypermodel.build(best_hp)
h_model.summary() 
history = h_model.fit(x_train_mod, y_train, epochs=10) 

# ================================================== Evaluation ======================================= 

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()

score = h_model.evaluate(x_test_mod, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

h_model.save('mnist.h5')
print("Saving the model as mnist.h5")
