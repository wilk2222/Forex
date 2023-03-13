from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K

class Keras_Perceptron:
    def __init__(self, input_shape, output_units):
        self.input_shape = input_shape
        self.output_units = output_units
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Input layer
        model.add(Dense(units=self.output_units, input_shape=(self.input_shape,)))
        model.add(Activation('sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, epochs, batch_size):
        # Train the model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    def predict(self, X_test):
        # Generate predictions using the trained model
        predictions = self.model.predict(X_test)
        return (predictions > 0.5).astype(int)
