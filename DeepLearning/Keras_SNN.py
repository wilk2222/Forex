from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer, Flatten, TimeDistributed
from keras.layers import LSTM, Conv2D, MaxPooling2D, Reshape, Activation
from keras.constraints import max_norm
from keras import backend as K
import numpy as np

class Keras_SNN:
    def __init__(self, input_shape, output_units):
        self.input_shape = input_shape
        self.output_units = output_units
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Input layer
        model.add(InputLayer(input_shape=self.input_shape))
        model.add(Reshape((28, 28, 1)))

        # Convolutional layer
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten output
        model.add(Flatten())

        # Spiking layers
        model.add(Dense(units=64, activation='relu', use_bias=False))
        model.add(Dropout(0.5))
        model.add(Dense(units=self.output_units, activation='softmax', use_bias=False))
        model.add(Activation('tanh'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, epochs, batch_size):
        # Convert input data to spike trains
        X_train = np.abs(np.random.normal(loc=X_train, scale=0.5)).astype(int)
        X_train[X_train > 1] = 1

        # Convert labels to one-hot vectors
        y_train = np.eye(self.output_units)[y_train]

        # Train the model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    def predict(self, X_test):
        # Convert input data to spike trains
        X_test = np.abs(np.random.normal(loc=X_test, scale=0.5)).astype(int)
        X_test[X_test > 1] = 1

        # Generate predictions using the trained model
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)
