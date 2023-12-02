"""Visualize the distribution of different samples."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(sample, title, bins=16, **kwargs):
    """Plot the histogram of a given sample of random values.

    Parameters
    ----------
    sample : pandas.Series
        raw values to build histogram
    title : str
        plot title/header
    bins : int
        number of bins in the histogram
    kwargs : dict
        any other keyword arguments for plotting (optional)
    """
    # TODO: Plot histogram

    # TODO: show the plot

    sample.hist(bins=bins, **kwargs)
    plt.title(title)
    plt.show()

    return


def plot_normal_distribution():
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html

    mu, sigma = 0, 0.1  # mean and standard deviation
    s = np.random.default_rng().normal(mu, sigma, 1000)

    abs(mu - np.mean(s))
    abs(sigma - np.std(s, ddof=1))

    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.show()


def plot_exponential_distribution():
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html

    s = np.random.default_rng().exponential(scale=1, size=1000)
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.show()


def plot_lognormal_distribution():
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html

    mu, sigma = 3., 1.  # mean and standard deviation
    s = np.random.lognormal(mu, sigma, 1000)

    count, bins, ignored = plt.hist(s, 100, density=True, align='mid')
    x = np.linspace(min(bins), max(bins), 10000)
    pdf = (np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))
           / (x * sigma * np.sqrt(2 * np.pi)))
    plt.plot(x, pdf, linewidth=2, color='r')
    plt.plot()
    plt.axis('tight')
    plt.title="lognormal"
    plt.show()


def plot_uniform_distribution():
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html

    s = np.random.uniform(-1, 0, 1000)

    np.all(s >= -1)
    np.all(s <= 0)

    count, bins, ignored = plt.hist(s, 15, density=True)
    plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.title = "Uniform"
    plt.show()


if __name__ == '__main__':
    plot_exponential_distribution()
    plot_normal_distribution()
    plot_uniform_distribution()
    plot_lognormal_distribution()