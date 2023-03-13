from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K

class Keras_MLP:
    def __init__(self, input_shape, output_units, hidden_layers=[32, 16], activation='relu'):
        self.input_shape = input_shape
        self.output_units = output_units
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Input layer
        model.add(Dense(units=self.hidden_layers[0], input_shape=(self.input_shape,)))
        model.add(Activation(self.activation))

        # Hidden layers
        for layer_size in self.hidden_layers[1:]:
            model.add(Dense(units=layer_size))
            model.add(Activation(self.activation))

        # Output layer
        model.add(Dense(units=self.output_units))
        model.add(Activation('softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, epochs, batch_size):
        # Train the model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    def predict(self, X_test):
        # Generate predictions using the trained model
        predictions = self.model.predict(X_test)
        return predictions.argmax(axis=-1)
