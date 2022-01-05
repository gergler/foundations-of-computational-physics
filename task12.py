import numpy as np
import matplotlib.pyplot as plt

a0, a1, w0, w1 = 1, 0.05, 5.1, 25.5
T = 2 * np.pi

N = 200
t = np.linspace(0, T, N)


def f(x):
    return a0 * np.sin(w0 * x) + a1 * np.sin(w1 * x)


def rectangular_h(k):
    if N > k >= 0:
        return 1
    return 0


def haan_h(k):
    if N > k >= 0:
        return (1 - np.cos(2 * np.pi * k / N)) / 2
    return 0


def fourier(h):
    w, pow_spect = [], []
    for i in range(0, round(N / 2)):
        fi = complex(0, 0)
        for k in range(0, N):
            fi += f(t[k]) * np.exp(2 * np.pi * 1j * i * k / N) * h(k)
        pow_spect.append((fi * fi.conjugate()))
        w.append(2 * np.pi * i / T)
    return w, pow_spect


def py_fourier():
    for i in range(0, round(N / 2)):
        fr = np.fft.fft(f(t))*rectangular_h(i)
    return fr * fr.conjugate()


def main():
    rect_spect = fourier(rectangular_h)
    haan_spect = fourier(haan_h)
    w, rect_intensity, haan_intensity = rect_spect[0], rect_spect[1], haan_spect[1]

    fig = plt.figure(figsize=(16, 10))
    plt_signal = fig.add_subplot(421)
    plt_rect = fig.add_subplot(422)
    plt_haan = fig.add_subplot(423)
    py_plt = fig.add_subplot(424)
    error_plt = fig.add_subplot(426)

    py_plt.set_title('Spectrum |f(w)**2|, rectangular window by PYTHON')
    py_plt.set_yscale('log')
    py_plt.plot(w, py_fourier()[:int(N/2)], c='c')
    py_plt.grid()
    py_plt.axvline(w0, c='k', linestyle='dashed')
    py_plt.axvline(w1, c='k', linestyle='dashed')
    py_plt.set_xlim(0, 50)
    py_plt.set_ylim(10 ** (-2), 10 ** 5)
    py_plt.set_xlabel('w')
    py_plt.set_ylabel('|f(w)|^2')

    error_plt.set_title('|f(w)**2| error')
    error_plt.plot(w, py_fourier()[:int(N / 2)] - rect_spect[1], c='y')
    error_plt.grid()
    error_plt.axvline(w0, c='k', linestyle='dashed')
    error_plt.axvline(w1, c='k', linestyle='dashed')
    error_plt.set_xlim(0, 50)
    error_plt.set_xlabel('w')

    plt_signal.set_xlabel('t')
    plt_signal.set_ylabel('f(t)')
    plt_signal.set_title(f'Signal f(t), N = {N}')
    plt_signal.plot(t, f(t), c='green')

    plt_rect.set_yscale('log')
    plt_rect.set_xlabel('w')
    plt_rect.set_ylabel('|f(w)|^2')
    plt_rect.set_title('Spectrum |f(w)**2|, rectangular window')
    plt_rect.plot(w[1:], rect_intensity[1:], c='darkorange')
    plt_rect.set_xlim(0, 50)
    plt_rect.set_ylim(10 ** (-2), 10 ** 5)

    plt_haan.set_yscale('log')
    plt_haan.set_xlabel('w')
    plt_haan.set_ylabel('|f(w)|^2')
    plt_haan.set_title('Spectrum |f(w)**2|, Haan window')
    plt_haan.plot(w[1:], haan_intensity[1:], c='red')
    plt_haan.set_xlim(0, 50)
    plt_haan.set_ylim(10 ** (-6), 10 ** 4)

    plt_signal.grid()

    plt_haan.grid()
    plt_haan.axvline(w0, c='k', linestyle='dashed')
    plt_haan.axvline(w1, c='k', linestyle='dashed')

    plt_rect.grid()
    plt_rect.axvline(w0, c='k', linestyle='dashed')
    plt_rect.axvline(w1, c='k', linestyle='dashed')

    fig.subplots_adjust(hspace=1.0, wspace=0.5)
    plt.show()


if __name__ == '__main__':
    main()
