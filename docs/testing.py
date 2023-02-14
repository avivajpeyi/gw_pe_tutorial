""" Plot a GW waveform time-domain and frequency domain given some parameters"""

import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform, get_fd_waveform


def get_waveform(
    mass1=10,
    mass2=10,
    spin1x=0,
    spin1y=0,
    spin1z=0,
    spin2x=0,
    spin2y=0,
    spin2z=0,
    eccentricity=0,
    mean_per_ano=0,
    lambda1=0,
    lambda2=0,
    # extrinsic
    long_asc_nodes=0,
    inclination=0,
    coa_phase=0,
    polarization=0,
    distance=100,
    delta_t=1.0/2048,
    f_lower=20,
    approximant="IMRPhenomPv2",
):
    hp, hc = get_td_waveform(**locals())
    return hp, hc

hp, hc = get_waveform(mass1=30, mass2=30)
plt.plot(hp.sample_times, hp)
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.show()
