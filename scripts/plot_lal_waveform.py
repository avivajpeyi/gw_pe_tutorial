import bilby
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.getLogger("bilby").setLevel(logging.ERROR)

OUTDIR = "outdir"


def make_waveform_generator(approximant="IMRPhenomPv2", duration=4, sampling_frequency=2048,
                 frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole, ):
    """Create a waveform generator"""
    generator_args = dict(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=frequency_domain_source_model,
            waveform_arguments=dict(
                waveform_approximant=approximant,
                reference_frequency=20.,
            )
        )
    return bilby.gw.WaveformGenerator(**generator_args)


def compute_waveform(waveform_generator, signal_parameters={}):
    """Compute the waveform"""
    parameters = dict(
        # two mass parameters
        mass_1=30, mass_2=30,
        # 6 spin parameters
        a_1=0.0, a_2=0.0, tilt_1=0, tilt_2=0, phi_jl=0,phi_12=0,
        # 2 tidal deformation parameters (for NS)
        lambda_1=0, lambda_2=0,
        # 7 extrinsic parameters (skyloc, timing, phase, etc)
        luminosity_distance=2000,
        theta_jn=0, ra=0, dec=0,
        psi=0, phase=0,
        geocent_time=0,
        )
    parameters.update(signal_parameters)
    h = waveform_generator.time_domain_strain(parameters)
    t = waveform_generator.time_array
    approximant = waveform_generator.waveform_arguments["waveform_approximant"]
    delta_t = 1./waveform_generator.sampling_frequency
    duration = waveform_generator.duration

    if "IMR" in approximant:
        # IMR templates the zero of time is at max amplitude (merger)
        # thus we roll the waveform back a bit
        for pol in h.keys():
            h[pol] = np.roll(h[pol], - len( h[pol])//3)

    h_phase = np.unwrap(np.arctan2(h['cross'], h['plus']))
    h_freq = np.diff(h_phase) / (2 * np.pi * delta_t)
    # nan the freqs after the merger
    h_freq[np.argmax(h_freq):] = np.nan
    return h['plus'], h_freq, t

def plot_waveform(waveform_generator, signal_parameters={}, fig=None):
    h_plus, h_freq, h_time = compute_waveform(waveform_generator, signal_parameters)
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    else:
        ax = fig.axes[0]
    ax.plot(h_time, h_plus, alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Strain")
    ax.set_xlim(min(h_time), max(h_time))
    # remove whitespace between subplots
    fig.subplots_adjust(hspace=0)
    return fig

def main():
    waveform_generator = make_waveform_generator(approximant="IMRPhenomPv2")
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    for mc in np.linspace(20, 60, 3):
        m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(mc, 1)
        signal_parameters = dict(mass_1=m1, mass_2=m2)
        plot_waveform(waveform_generator, signal_parameters, fig=fig)
    plt.show()
    fig.savefig('waveform_test.png')


if __name__ == '__main__':
    main()
