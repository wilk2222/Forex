from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, BatchNormalization

class Keras_RNN:
    def __init__(self, input_shape, output_units, hidden_layers, units, dropout_fraction):
        self.input_shape = input_shape
        self.output_units = output_units
        self.hidden_layers = hidden_layers
        self.units = units
        self.dropout_fraction = dropout_fraction
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        
        # Input layer
        model.add(SimpleRNN(units=self.units, return_sequences=True, input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_fraction))
        
        # Hidden layers
        for i in range(self.hidden_layers - 1):
            model.add(SimpleRNN(units=self.units, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_fraction))
        
        # Output layer
        model.add(SimpleRNN(units=self.output_units))
        model.add(Dense(units=self.output_units))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def fit(self, X_train, y_train, epochs, batch_size):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        return self.model.predict(X_test)
