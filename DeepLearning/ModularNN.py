from keras.models import Model
from keras.layers import Input, Dense, Dropout

class ModularNN:
    def __init__(self, input_shape, output_shape, hidden_layers):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = inputs

        for units, activation, dropout in self.hidden_layers:
            x = Dense(units, activation=activation)(x)
            if dropout > 0:
                x = Dropout(dropout)(x)

        outputs = Dense(self.output_shape, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def fit(self, X_train, y_train, epochs, batch_size):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions
