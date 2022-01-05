import numpy as np
import matplotlib.pyplot as plt

N = 3
L = 1
x0, xn = 0, L
h = (xn - x0) / N

T = 3  # точек во времени
t0, tn = 0, 1
tau = (tn - t0) / T

u0t = 0
uLt = 0


def ux0(x):
    return x * pow((1 - x / L), 2)


# коэффициенты 3д матрицы при решении зДирихле с нулевыми граничными условиями
def matrix():
    a, b, c = [], [], []

    for i in range(0, N - 1):
        a.append(-0.5 * tau / pow(h, 2))
        b.append(1 + tau / pow(h, 2))
        c.append(-0.5 * tau / pow(h, 2))

    a[0], c[N - 2] = 0, 0

    return a, b, c


def tridiagonal(d):
    a, b, c = matrix()

    for i in range(0, N - 1):
        ksi = a[i] / b[i - 1]
        b[i] -= ksi * c[i - 1]
        d[i] -= ksi * d[i - 1]

    y = [i for i in range(N - 1)]
    y[N - 2] = d[N - 2] / b[N - 2]

    for i in range(N - 3, -1, -1):
        y[i] = (d[i] - c[i] * y[i + 1]) / b[i]

    return y


def main():
    # значения x_j в начальный момент времени t=0
    u_x0j, x_j = [], []
    for j in range(0, N + 1):
        x_j.append(x0 + j * h)
        u_x0j.append(ux0(x_j[j]))

    u = [u_x0j]  # составляем список списков решений

    for m in range(T):
        d = []
        for j in range(1, N):
            d.append(u[m][j] + tau / 2 * (u[m][j + 1] - 2 * u[m][j] + u[m][j - 1]) / h ** 2)

        solution = tridiagonal(d)
        solution.insert(0, 0)
        solution.append(0)

        u.append(solution)

    u_max, t = [], []  # ищем максимальную температуру от времени

    for i in range(len(u)):
        u_max.append(max(u[i]))
        t.append(t0 + tau * i)

    plt.plot(t, u_max, color='green', label='u_max')
    plt.title('Зависимость максимальной температуры от времени')
    plt.xlabel('t')
    plt.ylabel('u_max')
    plt.show()


if __name__ == '__main__':
    main()
