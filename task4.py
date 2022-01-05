import numpy as np
import matplotlib.pyplot as plt

N = 100
h = 0.00001
step = 2 * np.pi / N

x_array, j0_array, j1_array, dj0_array, err_array = [], [], [], [], []

j = lambda x, t, m: (1 / np.pi) * np.cos(m * t - x * np.sin(t))
dj0 = lambda x: (simpson(x + h, 0) - simpson(x - h, 0)) / (2 * h)


def simpson(x, m):
    simps_int = 0
    t = np.linspace(0, np.pi, N + 1)
    step_size = t[1] - t[0]
    y = j(x, t, m)
    simps_int += np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])
    return simps_int * step_size / 3


def main():
    for i in range(N + 1):
        x_array.append(i * step)
        j0_array.append(simpson(i * step, 0))
        j1_array.append(simpson(i * step, 1))
        dj0_array.append(dj0(i * step))
        err_array.append(dj0_array[i] + j1_array[i])

    fig = plt.figure(figsize=(10, 4))
    plt_j = fig.add_subplot(121)
    plt_err = fig.add_subplot(122)
    plt_j.axis([0, 2 * np.pi, -0.75, 1.15])
    plt_j.plot(x_array, j0_array, color='purple', label='j0')
    plt_j.plot(x_array, j1_array, color='blue', label='j1')
    plt_j.set_xlabel('x - axis', labelpad=5, fontsize=12)
    plt_j.set_ylabel('y - axis', labelpad=5, fontsize=12)
    plt_j.set_title('Bessel')
    plt_j.legend()

    plt_err.axis([0, 2 * np.pi, -2.5e-11, 2.5e-11])
    plt_err.plot(x_array, err_array, color='pink', label='error')
    plt_err.set_xlabel('x - axis', labelpad=5, fontsize=12)
    plt_err.set_ylabel('y - axis', labelpad=5, fontsize=12)
    plt_err.set_title('Error')
    plt_err.legend()

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()


if __name__ == '__main__':
    main()
