import numpy as np
import matplotlib.pyplot as plt


def simulate_time_series(slope: float, baseline: int, amplitude: int, duration: int, plot: bool = False):
    time = np.arange(duration)
    series = baseline + trend(time, slope)
    if plot:
        plot_series(time, series)

    seasonality_pattern = seasonality(time, period=365, amplitude=amplitude)
    if plot:
        plot_series(time, seasonality_pattern)

    series += seasonality_pattern
    if plot:
        plot_series(time, series)

    noise_level = 5
    noise = white_noise(time, noise_level, seed=42)
    series += noise
    if plot:
        plot_series(time, series)
    return time, series


def plot_series(time, series, plot_format="-", start=0, end=None, label=None):
    plt.figure(figsize=(10, 6))
    plot_series_on_current_figure(time, series, plot_format, start, end, label)
    plt.show()


def plot_series_on_current_figure(time, series, plot_format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], plot_format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def trend(time: np.array, slope: float = 0):
    return slope * time


def seasonality(time: np.array, period: float, amplitude: int = 1, phase: int = 0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def seasonal_pattern(season_time: np.array):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level
