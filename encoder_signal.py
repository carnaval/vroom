import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter

def butter_lowpass_filter(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    print(f"cutoff = {normal_cutoff}")
    b, a = butter(order, normal_cutoff, btype='low')
    return lambda data: lfilter(b, a, data)
def iq_one_cycle_moving(signal, Fs, Fc):
    N = int(round(Fs / Fc))

    """
    n = np.arange(len(signal))
    phase = 2 * np.pi * Fc * n / Fs

    i_mixed = signal * np.cos(phase)
    q_mixed = signal * -np.sin(phase)

    kernel = np.ones(N) * (2 / N)

    I = np.convolve(i_mixed, kernel, mode="valid")
    Q = np.convolve(q_mixed, kernel, mode="valid")

    return I, Q
    """
    kernel = np.ones(N) * (2 / N)

    return np.convolve(signal, kernel, mode="same")

Fc = 500e3

if False:
    Ns = 128*1024
    T = 35.0 / Fc
    Fs = Ns / T
else:
    Fs = 9.0*Fc
    T = 135.0 / Fc
    Ns = int(round(T*Fs))
    print(Ns)
t = (np.arange(Ns) + 0.25) / Fs


i_mod = (np.sin(2.0 * np.pi * t * Fc))
q_mod = (np.sin(2.0 * np.pi *( t * Fc + 0.25)))

tx0 = np.sign(np.sin(2.0 * np.pi * t * Fc))
tx1 = np.sign(np.sin(2.0 * np.pi * (t * Fc + 1.0/3.0)))
tx2 = np.sign(np.sin(2.0 * np.pi * (t * Fc + 2.0/3.0)))


#plt.plot(t, tx0)
#plt.plot(t, tx2)
#plt.plot(t, tx3)

fig, (plt0, plt1) = plt.subplots(2)

adeg = np.piecewise(t, [t < T*0.5, t >= T*0.5], [40.0, 60.0])

rpm = 10000
deg_per_sec = rpm * 360 / 60

#adeg = np.linspace(0.0, deg_per_sec*T, Ns)
a = adeg * (2.0 * np.pi / 360.0)
w0 = np.cos(a)
w1 = np.cos(a + 2.0 * np.pi / 3.0)
w2 = np.cos(a + 4.0 * np.pi / 3.0)

sig = (w0 * tx0 + w1 * tx1 + w2 * tx2) / 3.0

i = sig * i_mod
q = sig * q_mod
plt0.plot(t, sig)

plt0.plot(t, i)
plt0.plot(t, q)

if True:
    i_lpf = iq_one_cycle_moving(i, Fs, Fc)
    q_lpf = iq_one_cycle_moving(q, Fs, Fc)

else:
    lpf = butter_lowpass_filter(Fc/10.0, Fs)
    i_lpf = lpf(i)
    q_lpf = lpf(q)



plt0.plot(t, i_lpf)
plt0.plot(t, q_lpf)

computed_adeg = np.arctan2(-q_lpf, i_lpf) * 360.0 / (2.0 * np.pi)
plt1.plot(t, adeg)
plt1.plot(t, computed_adeg)
plt1.plot(t, adeg - computed_adeg)

#plt1.plot(t, adeg - lpf(computed_adeg))

#plt1.plot(t, lpf(computed_adeg))
#plt1.plot(t, i_lpf * i_lpf + q_lpf * q_lpf)
plt.show()
