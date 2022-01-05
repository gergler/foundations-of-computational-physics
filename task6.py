import numpy as np
import matplotlib.pyplot as plt
import random

color = ['purple', 'blue', 'yellow', 'red', 'pink', 'green', 'orange']

N = 10

func = lambda x: -x
answer = lambda t: 1 / np.exp(t)

mint, maxt = 0, 3
x0, t0 = 1, 0
h = (maxt - mint) / N

alpha = 3 / 4  # optimal for O(h**3) / modified euler  = 1 (error lt) changed = 1/2


def broken_line():
    x, t = [x0], [t0]
    for i in range(N):
        x.append(x[i] + h * func(x[i]))
        t.append(t0 + h * i)
    return x, t


def runge_kutta_2_order():
    x, t = [x0], [t0]
    for i in range(N):
        x.append(x[i] + h * ((1 - alpha) * func(x[i]) + alpha * func(x[i] + h * func(x[i]) / (2 * alpha))))
        t.append(t0 + h * i)
    return x, t


def runge_kutta_4_order():
    x, t = [x0], [t0]
    for i in range(N):
        k1 = func(x[i])
        k2 = func(x[i] + h * k1 / 2)
        k3 = func(x[i] + h * k2 / 2)
        k4 = func(x[i] + h * k3)
        x.append(x[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        t.append(t0 + h * i)
    return x, t


def draw(plotter, name, data):
    plotter.set_title(name)
    col = random.choice(color)
    plotter.set_xlabel('x - axis')
    plotter.set_ylabel('t - axis')
    plotter.plot(data[1], data[0], color=col)


def main():
    t = np.linspace(mint, maxt, N)

    fig = plt.figure(figsize=(12, 8))

    plt_error = fig.add_subplot(212)
    plt_gr = fig.add_subplot(211)

    euler = broken_line()
    rk2 = runge_kutta_2_order()
    rk4 = runge_kutta_4_order()

    plt_gr.set_title('methods x(t)')
    plt_gr.set_xlabel('x')
    plt_gr.set_ylabel('t')
    plt_gr.plot(answer(t), t, color='red', label='analytics')
    plt_gr.plot(euler[0], euler[1], color='orange', label='euler')
    plt_gr.plot(rk2[0], rk2[1], color='green', label='rk2')
    plt_gr.plot(rk4[0], rk4[1], color='blue', label='rk4')

    plt_error.set_title('error x(t)')
    plt_error.set_xlabel('x')
    plt_error.set_ylabel('t')
    plt_error.plot(answer(t) - euler[0][1:], euler[1][1:], color='orange', label='euler')
    plt_error.plot(answer(t) - rk2[0][1:], rk2[1][1:], color='green', label='rk2')
    plt_error.plot(answer(t) - rk4[0][1:], rk4[1][1:], color='blue', label='rk4')

    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt_error.legend()
    plt_gr.legend()
    plt.show()


if __name__ == '__main__':
    main()
