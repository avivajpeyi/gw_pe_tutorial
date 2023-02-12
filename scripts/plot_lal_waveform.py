"""Module to plot Gravitational Wave Templates

Can be used to test if a waveform approximant works or not.

"""

from __future__ import division, print_function

import bilby
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)

OUTDIR = "outdir_waveform"

class Waveform:
    def __init__(self, approximant="IMRPhenomPv2", duration=4, sampling_frequency=2048,
                 frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole, post_merger_time=0.2):
        self.approximant = approximant
        generator_args = dict(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=frequency_domain_source_model,
            waveform_arguments=dict(
                waveform_approximant=approximant,
                reference_frequency=20.,
            )
        )
        self.duration = duration
        self.post_merger_time = post_merger_time
        self.delta_t = 1.0 / sampling_frequency
        self.generator = bilby.gw.WaveformGenerator(**generator_args)

    def __call__(self,
                 mass_1=30,
                 mass_2=30,
                 a_1=0.0,
                 a_2=0.0,
                 tilt_1=0,
                 tilt_2=0,
                 phi_jl=0,
                 phi_12=0,
                 luminosity_distance=2000,
                 theta_jn=0,
                 psi=0,
                 phase=0,
                 geocent_time=0,
                 ra=0,
                 dec=0,
                 lambda_1=0,
                 lambda_2=0,
                 ):
        parameters = {k: v for k, v in locals().items() if k not in ["self", 'parameters']}
        h = self.generator.time_domain_strain(parameters)
        t = self.generator.time_array

        if "IMR" in self.approximant:
            # IMR templates the zero of time is at max amplitude (merger)
            # thus we roll the waveform back a bit
            roll_back = int(self.post_merger_time / self.delta_t)
            for pol in h.keys():
                h[pol] = np.roll(h[pol], -roll_back)
            t0 = self.post_merger_time - self.duration
            t1 = self.post_merger_time
            t = np.arange(t0, t1, self.delta_t)
        h_phase = np.unwrap(np.arctan2(h['cross'], h['plus']))
        h_freq = np.diff(h_phase) / (2 * np.pi * self.delta_t)
        # nan the freqs after the merger
        h_freq[np.argmax(h_freq):] = np.nan
        return h['plus'], h_freq, t




def main():
    waveform = Waveform(approximant="IMRPhenomPv2")
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for mc in np.linspace(20, 60, 3):
        m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(mc, 1)
        h_plus, h_freq, h_time = waveform(mass_1=m1, mass_2=m2, luminosity_distance=1000)
        ax[0].plot(h_time, h_plus, alpha=0.5)
        ax[1].plot(h_time[:-1], h_freq, linewidth=3, alpha=0.5)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Strain")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Frequency (Hz)")
    ax[0].set_xlim(min(h_time), max(h_time))
    # remove whitespace between subplots
    fig.subplots_adjust(hspace=0)
    plt.show()

    fig.savefig('waveform_test.png')


if __name__ == '__main__':
    main()
