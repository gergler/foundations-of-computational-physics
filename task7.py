import matplotlib.pyplot as plt

a, b, c, d = 10, 2, 2, 10
alpha = 3 / 4
N = 1000

mint, maxt = 0, 5
h = (maxt - mint) / N

div_x = lambda x, y: a * x - b * x * y
div_y = lambda x, y: c * x * y - d * y

x0, y0, t0 = 1, 1, 0


def runge_kutta_2_order():
    x, y = [x0], [y0]
    for i in range(N + 1):
        x.append(x[i] + h * ((1 - alpha) * div_x(x[i], y[i]) + alpha * div_x(x[i] + h * div_x(x[i], y[i]) / (2 * alpha),
                                                                             y[i] + h * div_y(x[i], y[i]) / (2 * alpha))))
        y.append(y[i] + h * ((1 - alpha) * div_y(x[i], y[i]) + alpha * div_y(x[i] + h * div_x(x[i], y[i]) / (2 * alpha),
                                                                             y[i] + h * div_y(x[i], y[i]) / (2 * alpha))))
    return x, y


def main():
    rk2 = runge_kutta_2_order()
    plt.title('phase trajectory')
    plt.plot(rk2[0], rk2[1], color='green', label='y(x)')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == '__main__':
    main()
