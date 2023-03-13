from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization

class Keras_CNN1D:
    def __init__(self, input_shape, output_units, num_filters=32, filter_size=3, pool_size=2, hidden_layers=3, units=100, dropout_fraction=0.2):
        self.input_shape = input_shape
        self.output_units = output_units
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.hidden_layers = hidden_layers
        self.units = units
        self.dropout_fraction = dropout_fraction
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        
        # Input layer
        model.add(Conv1D(filters=self.num_filters, kernel_size=self.filter_size, activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(BatchNormalization())
        
        # Hidden layers
        for i in range(self.hidden_layers - 1):
            model.add(Conv1D(filters=self.num_filters, kernel_size=self.filter_size, activation='relu'))
            model.add(MaxPooling1D(pool_size=self.pool_size))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_fraction))
        
        # Output layer
        model.add(Flatten())
        model.add(Dense(units=self.units, activation='relu'))
        model.add(Dense(units=self.output_units))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def fit(self, X_train, y_train, epochs, batch_size):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        return self.model.predict(X_test)
