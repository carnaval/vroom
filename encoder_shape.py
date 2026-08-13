import matplotlib.pyplot as plt
import numpy as np
import sys

def conv_circ( signal, ker ):
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) )) / len(ker)

N = 16*3
assert(N % 6 == 0)
W = 2.0 * np.pi / 3.0
angle = np.linspace(0.0, 2.0 * np.pi, N)
rx_height = np.where(angle < np.pi, 1.0, 0.0)
"""
height_p1 = np.maximum(0.0, np.sin(angle + W*0))
height_p2 = np.maximum(0.0, np.sin(angle + W*1))
height_p3 = np.maximum(0.0, np.sin(angle + W*2))
"""
height_p1 = 1.0/3.0 * (1.0 - np.cos(angle + W*0))
height_p2 = 1.0/3.0 * (1.0 - np.cos(angle + W*1))
height_p3 = 1.0/3.0 * (1.0 - np.cos(angle + W*2))

normalizer = height_p1 + height_p2 + height_p3
assert(np.isclose(normalizer, 1.0).all())

# w = e^(i 2 pi / 3)

# cos(x) + w * cos(x + 2pi/3) + w^2 * cos(x + 4pi/3)
# e^ix (1 + w*w + w^2 * w^2) + e^(-ix) (1 + 1 + 1)
#
if False:
    for h in [height_p1, height_p2, height_p3]:
        plt.plot(angle, h)
area_p1 = conv_circ(rx_height, height_p1)
area_p2 = conv_circ(rx_height, height_p2)
area_p3 = conv_circ(rx_height, height_p3)
if False:
    for area in [area_p1, area_p2, area_p3]:
        plt.plot(angle, area)

    w = np.exp(1j*W)
    w0 = 1.0
    w1 = w
    w2 = w*w
    phasor = w0 * area_p1 + w1 * area_p2 + w2 * area_p3
    plt.plot(angle, np.angle(phasor))

#plt.show()
#sys.exit(0)

import numpy as np

def periodic_derivative_(y, dx):
    n = len(y)
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    return np.fft.ifft(1j * k * np.fft.fft(y)).real

def periodic_derivative(y, dx):
    return (np.roll(y, -1) - np.roll(y, 1)) / (2 * dx)

import numpy as np

def separated_curves_local(p, r, R, s0, dtheta):
    H = R - r

    c = H * p
    dc_dtheta = periodic_derivative(c, dtheta)

    rho = r + c

    g = s0 * np.sqrt(1.0 + (dc_dtheta / rho)**2)

    a = p * (H - g)
    b = a + g

    return a, b



# shape
total_height = 25e-3
inner_radius = 25e-3
track_spacing = 0.1e-3
outer_radius = inner_radius + total_height

#plt.plot(angle, height_p1)

low_0 = np.full(N//3, 0.0)
high_0 = height_p1[0:N//3]
low_1 = np.copy(high_0)
high_1 = height_p1[N//3:2*N//3] + low_1
low_2 = np.copy(high_1)
high_2 = height_p1[2*N//3:3*N//3] + low_2
low = np.concat((low_0, low_1, low_2))
high = np.concat((high_0, high_1, high_2))
#plt.plot(angle, low)
#plt.plot(angle, high)
curve_y =  np.concat((low, np.flip(high)))
curve_x = np.concat((angle, np.flip(angle)))

#plt.plot(angle, height_p2)
#plt.plot(angle, height_p3)
#plt.plot(angle, height_p1 + height_p2 + height_p3)

#curve = np.concat((height_p1[0:N//4], height_p1[0:N//4], np.full(N - N//2, 0.0)))
#curve_y = np.concat((height_p1[0:N//4], np.full(N//4, 1.0), np.flip(height_p1[0:N//4]), np.full(N//4, 0.0)))
#curve_x = np.concat((angle[0:N//2], np.flip(angle[0:N//2])))
#plt.plot(curve_x, curve_y)
if False:
    for da in [0.0, 2.0*np.pi/3, 4.0*np.pi/3]:
        plt.polar(curve_x + da, inner_radius + total_height * curve_y)


f = open('test.lst', 'w')
z = 0.0

def output_poly(f, name, low, high, z):
    mid = (low+high)*0.5
    low = low + (mid - low)*0.01
    high = high + (mid - high)*0.01
    low_cart = (np.cos(angle) * (inner_radius + total_height * low), np.sin(angle) * (inner_radius + total_height * low))
    high_cart = (np.cos(angle) * (inner_radius + total_height * high), np.sin(angle) * (inner_radius + total_height * high))
    output_poly_cart(f, name, low_cart, high_cart, z)
def output_poly_cart(f, name, low_cart, high_cart, z):
    for i in range(len(low)-1):
        l0 = (low_cart[0][i],low_cart[1][i])
        h0 = (high_cart[0][i],high_cart[1][i])
        l1 = (low_cart[0][i+1],low_cart[1][i+1])
        h1 = (high_cart[0][i+1], high_cart[1][i+1])
        if np.isclose(l0, h0).all():
            continue
        if np.isclose(l1, h1).all():
            continue
        f.write(f"Q {name} {l0[0]:.20f} {l0[1]:.20f} {z:.20f} {l1[0]:.20f} {l1[1]:.20f} {z:.20f} {h1[0]:.20f} {h1[1]:.20f} {z:.20f} {h0[0]:.20f} {h0[1]:.20f} {z:.20f}\n")

output_poly(f, "plate_1", low, high, 0.0)
output_poly(f, "plate_2", np.roll(low, N//3), np.roll(high, N//3), 0.0)
output_poly(f, "plate_3", np.roll(low, 2*N//3), np.roll(high, 2*N//3), 0.0)

x = np.linspace(-outer_radius, outer_radius, N)
output_poly_cart(f, "rotor_1", (x, np.full(N, 0.1e-3)), (x, np.sqrt(outer_radius*outer_radius - x*x) + 0.1e-3), 1e-3)
output_poly_cart(f, "rotor_2", (x, np.full(N, -0.1e-3)), (x, -np.sqrt(outer_radius*outer_radius - x*x) - 0.1e-3), 1e-3)
output_poly_cart(f, "rotor_1", (x, np.full(N, 0.1e-3)), (x, np.sqrt(outer_radius*outer_radius - x*x) + 0.1e-3), -1e-3)
output_poly_cart(f, "rotor_2", (x, np.full(N, -0.1e-3)), (x, -np.sqrt(outer_radius*outer_radius - x*x) - 0.1e-3), -1e-3)

f.close()

#plt.polar(angle, inner_radius + height_p1 * total_height)
#r = inner_radius + height_p1 * total_height
#plt.plot(np.cos(angle) * r, np.sin(angle) * r)
#
"""
(a,b) = separated_curves_local(height_p1, inner_radius, outer_radius, track_spacing, 2.0 * np.pi / N)
plt.plot(angle, inner_radius + a)
plt.plot(angle, inner_radius + b)
"""
"""
r = inner_radius + a
plt.plot(np.cos(angle) * r, np.sin(angle) * r)
r = inner_radius + b
plt.plot(np.cos(angle) * r, np.sin(angle) * r)
plt.axis('equal')
"""

plt.show()
