import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import random

color = ['purple', 'blue', 'yellow', 'red', 'pink']

func1 = lambda x: 1 / (1 + x ** 2)

func2 = lambda x: x ** (1 / 3.0) * np.exp(np.sin(x))


# слишком быстро меняется, проверка в WA с cos/(1)/(1/2)/(1/10)


def draw(func, a, b, N, x, name):
    X = np.linspace(a, b, 100)
    Y = np.vectorize(func)(X)

    col = random.choice(color)

    plt.plot(X, Y, color=col)

    for i in range(N):
        xs = [x[i], x[i], x[i + 1], x[i + 1]]
        ys = [0, func(x[i]), func(x[i + 1]), 0]
        plt.fill(xs, ys, edgecolor=col, alpha=0.2)

    plt.title(f'{name}, N = {N}')
    plt.show()


def trapezoid(func, a, b, N=10):
    x = np.linspace(a, b, N + 1)
    y = np.vectorize(func)(x)

    # draw(func, a, b, N, x, 'Trapezoid')

    return (b - a) / (2 * N) * np.sum(y[1:] + y[:-1])


def simpson(func, a, b, N=10):
    x = np.linspace(a, b, N + 1)
    y = np.vectorize(func)(x)

    # draw(func, a, b, N, x, 'Simpson')

    return (b - a) / (N * 3) * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])


def main(func, a, b, N):
    integral = quad(func, a, b)[0]
    for i in range(1, N):
        n = 2 ** i
        T = trapezoid(func, a, b, n)
        err_t = abs(T - integral)
        S = simpson(func, a, b, n)
        err_s = abs(S - integral)
        print(f'N: {n}\t|\t Trapezoid: {T} \t|\t T_error: {err_t} \t|\t Simpson: {S} \t|\t S_error: {err_s} ')


if __name__ == '__main__':
    # main(func1, -1, 1, 2)
    main(func2, 0.1, 1, 10)
