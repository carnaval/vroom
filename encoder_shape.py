import matplotlib.pyplot as plt
import numpy as np

def conv_circ( signal, ker ):
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) )) / len(ker)

N = 1024
W = 2.0 * np.pi / 3.0
angle = np.linspace(0.0, 2.0 * np.pi, N)
rx_height = np.where(angle < np.pi, 1.0, 0.0)
height_p1 = np.maximum(0.0, np.cos(angle + W*0))
height_p2 = np.maximum(0.0, np.cos(angle + W*1))
height_p3 = np.maximum(0.0, np.cos(angle + W*2))

# w = e^(i 2 pi / 3)

# cos(x) + w * cos(x + 2pi/3) + w^2 * cos(x + 4pi/3)
# e^ix (1 + w*w + w^2 * w^2) + e^(-ix) (1 + 1 + 1)
#
for h in [height_p1, height_p2, height_p3]:
    plt.plot(angle, h)
area_p1 = conv_circ(rx_height, height_p1)
area_p2 = conv_circ(rx_height, height_p2)
area_p3 = conv_circ(rx_height, height_p3)
for area in [area_p1, area_p2, area_p3]:
    plt.plot(angle, area)
w = np.exp(1j*W)
w0 = 1.0
w1 = w
w2 = w*w
phasor = w0 * area_p1 + w1 * area_p2 + w2 * area_p3
plt.plot(angle, np.angle(phasor))

plt.ylabel('some numbers')
plt.show()
