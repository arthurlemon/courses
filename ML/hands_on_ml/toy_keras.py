import keras_tuner as kt
import tensorflow as tf
from loguru import logger


def pipeline_mnist():
    data = tf.keras.datasets.fashion_mnist.load_data()
    (X_train_full, y_train_full), (X_test, y_test) = data
    X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
    X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

    tf.random.set_seed(42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    logger.info(model.summary())


    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=30,
            validation_data=(X_valid, y_valid),
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
            batch_size=32)

    model.evaluate(X_test, y_test)

def pipeline_california_houses():
    data = tf.keras.datasets.boston_housing.load_data()
    (X_train_full, y_train_full), (X_test, y_test) = data
    X_train, y_train = X_train_full[:-100], y_train_full[:-100]
    X_valid, y_valid = X_train_full[-100:], y_train_full[-100:]

    tf.random.set_seed(42)

    norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
    model = tf.keras.models.Sequential([
        norm_layer,
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss="mse",
                  metrics=["RootMeanSquaredError"],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    norm_layer.adapt(X_train)
    logger.info(model.summary())
    model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
    model.evaluate(X_test, y_test)

def functional_keras_api():
    data = tf.keras.datasets.boston_housing.load_data()
    (X_train_full, y_train_full), (X_test, y_test) = data
    X_train, y_train = X_train_full[:-100], y_train_full[:-100]
    X_valid, y_valid = X_train_full[-100:], y_train_full[-100:]

    X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
    X_valid_wide, X_valid_deep = X_valid[:, :5], X_valid[:, 2:]
    X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
    X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

    # architecture 2
    input_wide = tf.keras.layers.Input(shape=[5])
    input_dense = tf.keras.layers.Input(shape=[11])

    norm_layer_wide = tf.keras.layers.Normalization()
    norm_wide = norm_layer_wide(input_wide)
    norm_layer_deep = tf.keras.layers.Normalization()
    norm_deep = norm_layer_deep(input_dense)
    hidden1 = tf.keras.layers.Dense(30, activation='relu')(norm_deep)
    hidden2 = tf.keras.layers.Dense(30, activation='relu')(hidden1)
    concat = tf.keras.layers.Concatenate()([norm_wide, hidden2])
    output = tf.keras.layers.Dense(1)(concat)
    model = tf.keras.Model(inputs=[input_wide, input_dense], outputs=[output])

    # the rest is the same
    model.compile(loss="mse",
                  optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3),
                  metrics=["RootMeanSquaredError"])

    logger.info(model.summary())

    norm_layer_wide.adapt(X_train_wide)
    norm_layer_deep.adapt(X_train_deep)

    tensorboard_cb = tf.keras.callbacks.TensorBoard("./ML/hands_on_ml/tensorboard_logs/functional_model",
                                                    profile_batch=(100, 200))

    history = model.fit((X_train_wide, X_train_deep), y_train, epochs=20, validation_data=((X_valid_wide,X_valid_deep), y_valid),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                         tf.keras.callbacks.ModelCheckpoint("./ML/hands_on_ml/model/functional_model",
                                                            save_weights_only=True,
                                                            save_best_only=True),
                                                            tensorboard_cb])
    logger.info(model.evaluate((X_test_wide, X_test_deep), y_test))

    model.save("./ML/hands_on_ml/model/functional_model", save_format="tf")

    model = tf.keras.models.load_model("./ML/hands_on_ml/model/functional_model")

def class_keras_api():
    """Toy example using the Keras API with a custom model class."""

    class WideAndDeepModel(tf.keras.Model):
        def __init__(self, units=30, activation='relu', **kwargs):
            super().__init__(**kwargs)
            self.norm_layer_wide = tf.keras.layers.Normalization()
            self.norm_layer_deep = tf.keras.layers.Normalization()
            self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
            self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
            self.main_output = tf.keras.layers.Dense(1)
            self.aux_output = tf.keras.layers.Dense(1)

        def call(self, inputs):
            input_wide, input_deep = inputs
            norm_wide = self.norm_layer_wide(input_wide)
            norm_deep = self.norm_layer_deep(input_deep)
            hidden1 = self.hidden1(norm_deep)
            hidden2 = self.hidden2(hidden1)
            concat = tf.keras.layers.concatenate([norm_wide, hidden2])
            main_output = self.main_output(concat)
            aux_output = self.aux_output(hidden2)
            return main_output, aux_output

    model = WideAndDeepModel(30, activation='relu', name="custom_model")
    model.save("my_custom_model", save_format="tf")

def keras_tuner():
    # option 1 - Use SciKeras wrapper for compatibility with scikit-learn optimizers/pipelines
    # option 2 - Use KerasTuner API

    def build_model(hp):
        n_hidden = hp.Int('n_hidden', min_value=0, max_value=8, default=2)
        n_neurons = hp.Int('n_neurons', min_value=16, max_value=256)
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        optimizer = hp.Choice('optimizer', values=['adam', 'sgd'])
        optimizer = tf.keras.optimizers.get(optimizer,learning_rate=learning_rate)

        model = tf.keras.Sequential(
            [tf.keras.layers.Flatten()])
        for _ in range(n_hidden):
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model

    random_search_tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=5,
                                          overwrite=True, directory="./ML/hands_on_ml/tuner_logs", seed=42)
    # random_search_tuner.search(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))
    # best_models = random_search_tuner.get_best_models(num_models=2)

def usual_batch_normalization():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
        tf.keras.layers.BatchNormalization()),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])


def transfer_learning():
    """Transfer learning with Keras"""

    model_A = tf.keras.models.load_model("model_A")
    model_A_clone = tf.keras.models.clone_model(model_A)
    model_A_clone.set_weights(model_A.get_weights()) # needed to reuse the same weights
    model_B_on_A = tf.keras.models.Sequential(model_A_clone.layers[:-1])
    model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    for layer in model_B_on_A.layers[:-1]:
        layer.trainable = False # to avoid overwriting the weights of the pretrained model in the first few epochs

    optimizer = tf.keras.optimizers.SGD(lr=1e-3)
    model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]) # compile after (un)freezing layers

    # first pass to roughly train new layers
    history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4, validation_data=(X_valid_B, y_valid_B))

    # now unfreeze the pretrained layers and continue training
    for layer in model_B_on_A.layers[:-1]:
        layer.trainable = True

    model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16, validation_data=(X_valid_B, y_valid_B))

def refactor_keras_code():

    from functools import partial
    RegularDense = partial(tf.keras.layers.Dense, activation="relu",
                            kernel_initializer="he_normal",
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        RegularDense(300),
        RegularDense(100),
        RegularDense(10, activation="softmax")
    ])

def low_level_api():
    # custom layers in Keras

    exp_layer = tf.keras.layers.Lambda(lambda x: tf.exp(x)) # layer without weights, always use tf vectorized functions

    # custom layer, inherit Layer class (similar to Model class for Subclassing API)
    class MyDense(tf.keras.layers.Layer):
        def __init__(self, units, activation=None, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.activation = tf.keras.activations.get(activation)

        def build(self, batch_input_shape):
            self.kernel = self.add_weight(name="kernel", shape=[batch_input_shape[-1], self.units],
                                            initializer="glorot_normal")
            self.bias = self.add_weight(name="bias", shape=[self.units], initializer="zeros")


        def call(self, X):
            return self.activation(X @ self.kernel + self.bias)

        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "units": self.units, "activation": tf.keras.activations.serialize(self.activation)}








if __name__ == '__main__':
    functional_keras_api()
