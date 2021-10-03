import tensorflow as tf
keras = tf.keras


class SimpleRnn:

    @staticmethod
    def build(neurons_per_mem_cell: int, amplitude: float):
        model = keras.models.Sequential([
            keras.layers.SimpleRNN(neurons_per_mem_cell, return_sequences=True),
            keras.layers.SimpleRNN(neurons_per_mem_cell, return_sequences=True),
            keras.layers.Dense(1),
            keras.layers.Lambda(lambda x: x * amplitude)
        ])

        optimizer = keras.optimizers.SGD(lr=1.5e-6, momentum=0.9)
        model.compile(loss=keras.losses.Huber(),
                      optimizer=optimizer,
                      metrics=["mae"])
        return model

