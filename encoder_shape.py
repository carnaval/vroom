import matplotlib.pyplot as plt
import numpy as np
import sys

def conv_circ( signal, ker ):
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) )) / len(ker)

N = 32*16*3
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
if True:
    for da in [0.0, 2.0*np.pi/3, 4.0*np.pi/3]:
        plt.polar(curve_x + da, inner_radius + total_height * curve_y)


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


# gen footprint
f = open('test.kicad_mod', 'w')

points = ""
for (x,y) in zip(curve_x, curve_y):
    r = inner_radius + total_height * y
    (x,y) = (np.cos(x)*r, np.sin(x)*r)
    x_mm = x*1e3
    y_mm = y*1e3
    points += f"(xy {x_mm} {y_mm})"

primitives = f"""
(gr_poly
	(pts {points}
	)
	(width 0.0)
	(fill yes)
)"""

footprint_text = f"""
(footprint "StatorTx"
	(version 20260206)
	(generator "me")
	(generator_version "10.0")
	(layer "F.Cu")
	(property "Reference" "REF**"
		(at 0 -0.5 0)
		(unlocked yes)
		(layer "F.SilkS")
		(uuid "aad3c48b-ea91-4c02-ab62-e1e8f505299c")
		(effects
			(font
				(size 1 1)
				(thickness 0.1)
			)
		)
	)
	(property "Value" "Untitled"
		(at 0 1 0)
		(unlocked yes)
		(layer "F.Fab")
		(uuid "c33fb2a9-4585-48ac-aef3-23aa1b2cc5e5")
		(effects
			(font
				(size 1 1)
				(thickness 0.15)
			)
		)
	)
	(property "Datasheet" ""
		(at 0 0 0)
		(unlocked yes)
		(layer "F.Fab")
		(hide yes)
		(uuid "8db52979-e032-42c2-9101-c0ed8832c9d7")
		(effects
			(font
				(size 1 1)
				(thickness 0.15)
			)
		)
	)
	(property "Description" ""
		(at 0 0 0)
		(unlocked yes)
		(layer "F.Fab")
		(hide yes)
		(uuid "376ad0a7-2890-4bcd-be53-35fabd68095b")
		(effects
			(font
				(size 1 1)
				(thickness 0.15)
			)
		)
	)
	(attr smd)
	(duplicate_pad_numbers_are_jumpers no)
	(fp_text user "${{REFERENCE}}"
		(at 0 2.5 0)
		(unlocked yes)
		(layer "F.Fab")
		(uuid "5027c18d-1e68-4ed7-95f0-973edbbfdf72")
		(effects
			(font
				(size 1 1)
				(thickness 0.15)
			)
		)
	)
	(pad "2" smd custom
		(at 0.0 0.0)
		(size 6.5675 6.5675)
		(layers "F.Cu" "F.Mask" "F.Paste")
		(options
			(clearance outline)
			(anchor rect)
		)
		(primitives
		    {primitives}
		)
		(uuid "2762c063-429e-48b9-a0e8-d4ad42257a20")
	)
	(embedded_fonts no)
)
"""
f.write(footprint_text)
f.close()
