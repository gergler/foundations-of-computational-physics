import numpy as np
import matplotlib.pyplot as plt

y0, yn = 0, 0  # граничные условия
x0, xn = 0, np.pi  # промежуток

N = 10  # кол-во интервалов
h = (xn - x0) / N  # шаг

a0, b0, c0 = 0, 1, 0


def func(x):
    return np.sin(x)


def matrix():
    a, b, c, d, x = [a0], [b0], [c0], [], []

    for i in range(1, N):
        a.append(1)
        b.append(-2)
        c.append(1)

    a.append(0)
    b.append(1)
    c.append(0)
    d.append(y0)
    x.append(x0)

    for i in range(1, N):
        xi = x0 + h * i
        d.append(pow(h, 2) * func(xi))
        print(f'sinx: {func(xi)}')
        x.append(xi)

    d.append(yn)
    x.append(xn)

    print(f'd matrix: {d}')
    print(f'x: {x}')

    return a, b, c, d, x


def tridiagonal(a, b, c, d):
    # прямой ход метода Гаусса - исключение поддиагональных элементов ai, i =2,..,n
    for i in range(1, N + 1):
        ksi = a[i] / b[i - 1]
        b[i] -= ksi * c[i - 1]
        d[i] -= ksi * d[i - 1]

    print(f'b: {b}')
    print(f'd: {d}')

    # reverse
    y = [i for i in range(N + 1)]
    y[N] = d[N] / b[N]

    for i in range(N - 1, -1, -1):
        y[i] = (d[i] - c[i] * y[i + 1]) / b[i]

    print(f'y: {y}')

    return y


def solution(x):
    c1 = (y0 - yn + func(x0) - func(xn)) / (x0 - xn)
    c2 = y0 + func(x0) - c1 * x0
    return -func(x) + c1 * x + c2


def main():
    a, b, c, d, x = matrix()
    y = tridiagonal(a, b, c, d)
    sol = [solution(i) for i in x]
    error = [(y[i] - sol[i]) for i in range(len(x))]

    fig = plt.figure(figsize=(10, 4))

    plt_gr = fig.add_subplot(121)
    plt_err = fig.add_subplot(122)

    plt_gr.set_title('y(x)')
    plt_gr.set_xlabel('x')
    plt_err.set_ylabel('y')
    plt_gr.plot(x, y, color='purple', label='y(x)')
    plt_gr.plot(x, sol, color='blue', label='solution y(x)', linestyle='dashed')

    plt_err.set_title('error')
    plt_err.set_xlabel('x')
    plt_err.set_ylabel('y - solution')
    plt_err.plot(x, error, color='pink', label='error')

    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt_gr.legend()
    plt_err.legend()
    plt.show()


if __name__ == '__main__':
    main()
