from dl.rnn.lstm_rnn import LSTMRnn
from dl.rnn.reset_states_callback import ResetStatesCallback
from dl.rnn.simple_rnn import SimpleRnn
from dl.rnn.time_series_simulation import simulate_time_series, plot_series_on_current_figure
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
keras = tf.keras

model_name = 'lstm'


def main():
    time, series = simulate_time_series(slope=0.05, baseline=10, amplitude=40, duration=(4 * 365 + 1))
    split_time = 1000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    window_size = 30

    if model_name == 'simple':
        train_set = seq2seq_window_dataset(x_train, window_size, batch_size=128)
        valid_set = seq2seq_window_dataset(x_valid, window_size, batch_size=128)
        model = SimpleRnn.build(100, 200)

    else:
        train_set = sequential_window_dataset(x_train, window_size)
        valid_set = sequential_window_dataset(x_valid, window_size)
        model = LSTMRnn.build(100, 200)

    early_stopping = keras.callbacks.EarlyStopping(patience=50)
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        "my_checkpoint", save_best_only=True)
    reset_states = ResetStatesCallback()

    # lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-7 * 10 ** (epoch / 20))

    model.fit(train_set, epochs=500,
              validation_data=valid_set,
              callbacks=[early_stopping, model_checkpoint, reset_states])

    model = keras.models.load_model("my_checkpoint")

    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

    plt.figure(figsize=(10, 6))
    plot_series_on_current_figure(time_valid, x_valid)
    plot_series_on_current_figure(time_valid, rnn_forecast)
    plt.show()

    print(keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())


def window_dataset(series,
                   window_size,
                   batch_size=32,
                   shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def seq2seq_window_dataset(series, window_size, batch_size=32,
                           shuffle_buffer=1000):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def sequential_window_dataset(series, window_size):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda window: (window[:-1], window[1:]))
    return ds.batch(1).prefetch(1)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


if __name__ == '__main__':
    main()
