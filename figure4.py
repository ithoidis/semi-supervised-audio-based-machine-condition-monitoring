import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.style.use('seaborn-deep')

model = torch.load('./results/CNN1D_Model1/model_epoch9.pt')  # model 3

w = model.conv1.weight.data.cpu().numpy()
fs= 16000
nfilt = 6000
from matplotlib.ticker import NullFormatter
from scipy.io import loadmat
h_mic1 = loadmat('h_mic1.mat')['h_mic1']
h_mic1 = h_mic1.swapaxes(0, 1).swapaxes(1, 2).swapaxes(0, 1)
# azimuth, microphones, samples

for j in range(32):
    ir = np.empty((45, 8, nfilt))
    for az in range(45):
        for channel in range(8):
            ir[az, channel] = np.convolve(h_mic1[az, channel], w[j, channel])[40:-40]

    # get DFT
    fr = np.fft.fft(ir, axis=-1)[:, :, nfilt//2-1:]

    # ir_c = ir[:, 0, :] + ir[:, 4, :] + ir[:, 6, :] + ir[]
    plt.figure(1)
    # plt.title('%d-th kernel response' % j)
    plt.subplot(4, 8, j+1)
    fr_c = np.fft.fft(ir.sum(axis=1), axis=-1)[:, nfilt//2-1:]
    plt.imshow(20 * np.log10(np.abs(fr_c.transpose())+1), aspect='auto', origin='lower')

    if False:
        plt.figure(figsize=(9, 4))
        plt.title('Filter %d' % j)
        plt.subplot(121)
        [plt.magnitude_spectrum(w[j, i], fs) for i in range(8)]
        plt.subplot(122)
        [plt.phase_spectrum(w[j, i], fs) for i in range(8)]

    plt.figure(2)
    ax = plt.subplot(4, 8, j+1,  projection='polar')
    polar_r = 10*np.log10((np.abs(fr_c)**2).sum(axis=1))
    polar_r = np.append(polar_r, polar_r[0])
    ax.plot(np.arange(0, 360+1, 8) * 2 * np.pi / 360, polar_r)
    ax.set_rlim(np.amin(polar_r),np.amax(polar_r))

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    ax.grid(True)
    for r_label in ax.get_yticklabels():
        r_label.set_text('')
    for r_label in ax.get_xticklabels():
        r_label.set_text('')
    # ax.set_yticks([])
    # ax.set_xticks([])

    # ax.xaxis.set_ticks_position('none')


    #np.corrcoef(w[0, 1], w[0, 7])


plt.show()

