import numpy as np
import matplotlib.pyplot as plt

N = 16

func_x = lambda x: 1 + x / N
func_y = lambda x: np.log(x)


def divided_difference(x, y):
    for i in range(1, N + 1):
        y[i:] = (y[i:] - y[i - 1]) / (x[i:] - x[i - 1])
    return y


def newton(x_array, y_array, x_point):
    coefficient = divided_difference(x_array, y_array)
    polynomial = coefficient[N]
    for i in range(1, N + 1):
        polynomial = coefficient[N - i] + (x_point - x_array[N - i]) * polynomial
    return polynomial


def draw(x_func, x_data, y_data):
    fig = plt.figure(figsize=(10, 4))
    plt_x = fig.add_subplot(121)
    plt_n = fig.add_subplot(122)
    plt_x.set_title('error')
    plt_x.set_xlabel('x - axis')
    plt_x.set_ylabel('y - axis')
    plt_x.plot(x_data, (func_y(x_data) - y_data), color='purple', label='error')
    plt_x.axis([0.98, 2.02, -0.00005, 0.00005])

    plt_n.set_xlabel('x - axis')
    plt_n.set_ylabel('y - axis')
    plt_n.scatter(x_func, func_y(x_func), color='pink', label='y = ln(x)')
    plt_n.plot(x_data, y_data, color='purple', label=f'Pn(x), n = {N}')
    plt_n.legend()
    plt_n.axis([0.98, 2.02, -0.05, 0.75])

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()


def main():
    x = np.arange(0, N + 1)
    x_func = func_x(x)
    x_data = np.arange(0.5, 2.5, 0.001)
    y_data = []
    for i in range(len(x_data)):
        y_data.append(newton(x_func, func_y(x_func), x_data[i]))
    draw(x_func, x_data, y_data)


if __name__ == '__main__':
    main()
