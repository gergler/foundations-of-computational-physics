import numpy as np
import matplotlib.pyplot as plt

const = 0.2848
M, A, U0 = 1, 1, 10  # MeV, A, eV
parameter = 2 * M * pow(A, 2) * U0 * const
tolerance = 1e-04


def func(x):
    return 1 / np.tan(np.sqrt(parameter * (1 - x))) - np.sqrt(1 / x - 1)


def dfunc(x):
    return parameter * 1 / pow(np.sin(np.sqrt(parameter * (1 - x))), 2) / (2 * np.sqrt(parameter * (1 - x))) + 1 / (
            2 * np.sqrt(1 / x - 1) * pow(x, 2))


func_half, half = [], []


def dichotomy(a, b):
    i = len(half)
    func_a = func(a)
    func_b = func(b)
    half.append((a + b) / 2)
    func_half.append(func(half[i]))
    if np.abs(func_half[i]) < tolerance:
        return func_half, half
    elif np.sign(func_a) == np.sign(func_half[i]):
        return dichotomy(half[i], b)
    elif np.sign(func_b) == np.sign(func_half[i]):
        return dichotomy(a, half[i])


def iteration(x0):
    x_next_points, x0_points = [], []
    iter_lambda = 0.1
    x_next = x0
    x0 -= iter_lambda * np.sign(dfunc(x0)) * func(x0)
    while np.abs(x_next - x0) > tolerance:
        x_next = x0
        x0 -= iter_lambda * np.sign(dfunc(x0)) * func(x0)
        x0_points.append(x0)
        x_next_points.append(x_next)
    return x0_points, x_next_points


def newton(x0):
    x_next_points, x0_points = [], []
    x_next = x0
    new_lambda = 1.0 / dfunc(x0)
    x0 -= new_lambda * func(x0)
    while np.abs(x_next - x0) > tolerance:
        x_next = x0
        new_lambda = 1.0 / dfunc(x0)
        x0 -= new_lambda * func(x0)
        x0_points.append(x0)
        x_next_points.append(x_next)
    return x0_points, x_next_points


def ground_state(a, b):
    x_arr = []
    for x in np.arange(a, b, tolerance):
        if func(x) > 0:
            x_arr.append(x)
    return x_arr


def subplot_draw(val0, val1, color, name, plt, y):
    plt.plot(val0, val1, color=color)
    plt.scatter(val0, val1, color='black')
    plt.set_title(name)
    plt.set_xlabel('x')
    plt.set_ylabel(y)
    plt.plot(val0[-1], val1[-1], '*', color='orange', markersize=10)


def draw():
    a = 0.99
    b = 1 - tolerance
    x = ground_state(a, b)[-1]

    print(f"Ground state Energy level x = {x}: {a} < x < {b}\n")

    dic = dichotomy(0.1, b)
    print(f'D) steps: {len(dic[1])}, tolerance: {tolerance}, answer: {dic[1][-1]}')

    iter = iteration(1.99)
    print(f'I) steps: {len(iter[1])}, tolerance: {tolerance}, answer: {iter[1][-1]}')

    newt = newton(1.99)
    print(f'N) steps: {len(newt[1])}, tolerance: {tolerance}, answer: {newt[1][-1]}')

    fig = plt.figure(figsize=(16, 4))
    plt_dic = fig.add_subplot(131)
    plt_dic.axis([0.75, 0.877, -0.15, 0.5])
    plt_iter = fig.add_subplot(132)
    plt_iter.axis([0.75, 0.8, 0.77, 0.795])
    plt_new = fig.add_subplot(133)
    plt_new.axis([0.75, 0.8, 0.77, 0.795])

    subplot_draw(dic[1], dic[0], 'purple', 'dichotomy', plt_dic, 'f(x)')
    subplot_draw(iter[1], iter[0], 'yellow', 'iteration', plt_iter, 'x + lamb*f(x)')
    subplot_draw(newt[1], newt[0], 'pink', 'newton', plt_new, 'x + f(x)/f\'(x)')

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()


draw()
