import numpy as np
import matplotlib.pyplot as plt

import bilby

from bilby import run_sampler

from bilby.core.prior import Constraint, Uniform

from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters,
    generate_all_bbh_parameters
)
from bilby.gw.detector.networks import InterferometerList
from bilby.gw.detector.psd import PowerSpectralDensity
from bilby.gw.likelihood import GravitationalWaveTransient
from bilby.gw.prior import BBHPriorDict
from bilby.gw.result import CBCResult
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.utils import get_event_time
from bilby.gw.waveform_generator import WaveformGenerator
from gwpy.plot import Plot as GWpyPlot
from gwpy.timeseries import TimeSeries
import os

OUTDIR = "outdir"
os.makedirs(OUTDIR, exist_ok=True)

interferometers = InterferometerList(["H1", "L1"])
trigger_time = get_event_time("GW150914")
tc = trigger_time
start_time = trigger_time - 3
duration = 4
end_time = start_time + duration
roll_off = 0.2

# Get raw data
raw_data = {}
for interferometer in interferometers:
    print(
        "Getting analysis segment data for {}".format(interferometer.name)
    )
    analysis_data = TimeSeries.fetch_open_data(
        interferometer.name, start_time, end_time
    )
    interferometer.strain_data.roll_off = roll_off
    interferometer.strain_data.set_from_gwpy_timeseries(analysis_data)
    raw_data[interferometer.name] = analysis_data

# plot raw data:
plot = GWpyPlot(figsize=(12, 4.8))
ax = plot.add_subplot(xscale='auto-gps')
for ifo_name, data in raw_data.items():
    ax.plot(data, label=ifo_name)
ax.set_epoch(tc)
ax.set_xlim(tc-0.4, tc+0.2)
ax.set_ylabel('Strain noise')
ax.legend()
plot.show()

# recall Dan Browns talk -- keep only data in 50-250 Hz and remove 60 Hz and 120 Hz (the 'violin modes')
# plot of Noise
# plot raw data after some basic filtering
plot = GWpyPlot(figsize=(12, 4.8))
ax = plot.add_subplot(xscale='auto-gps')
for ifo_name, data in raw_data.items():
    filtered_data = data.bandpass(50, 250).notch(60).notch(120)
    ax.plot(filtered_data, label=ifo_name)
ax.set_epoch(tc)
ax.set_xlim(tc-0.4, tc+0.2)
ax.set_ylim(-1e-21, 1e-21)
ax.set_ylabel('Strain noise')
ax.legend()
plot.show()

# Get data for noise estimation -- the power spectral density (PSD)
psd_start_time = start_time + duration
psd_duration = 128
psd_end_time = psd_start_time + psd_duration
psd_tukey_alpha = 2 * roll_off / duration
overlap = duration / 2

for interferometer in interferometers:
    print("Getting psd segment data for {}".format(interferometer.name))
    psd_data = TimeSeries.fetch_open_data(
        interferometer.name, psd_start_time, psd_end_time
    )
    psd = psd_data.psd(
        fftlength=duration, overlap=overlap, window=("tukey", psd_tukey_alpha),
        method="median"
    )
    interferometer.power_spectral_density = PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )

for interferometer in interferometers:
    analysis_data = abs(interferometer.frequency_domain_strain)
    plt.loglog(interferometer.frequency_array, analysis_data, label="Analysis Data")
    plt.loglog(interferometer.frequency_array, abs(interferometer.amplitude_spectral_density_array),
               label="ASD (estimated noise)")
    plt.xlim(interferometer.minimum_frequency, interferometer.maximum_frequency)
    ymin_max = [min(analysis_data), max(analysis_data)]
    plt.vlines([60, 120], *ymin_max, ls="--", color='k', zorder=-10)
    plt.fill_betweenx(ymin_max, 50, 250, color='tab:green', alpha=0.1)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r'Strain [strain/$\sqrt{\rm Hz}$]')
    plt.title(f"{interferometer.name} data")
    plt.ylim(*ymin_max)
    plt.legend()
    plt.show()

tc = trigger_time

for ifo_name, data in raw_data.items():
    qtrans = data.q_transform(
        frange=(20,500), fres = 0.05,
        outseg=(tc-0.2, tc+0.1)
    )
    plot = qtrans.plot(cmap = 'viridis', dpi = 150)
    ax = plot.gca()
    ax.set_title(f'{ifo_name} Q-transform')
    ax.set_epoch(trigger_time)
    ax.set_yscale('log')
    ax.colorbar(label="Normalised energy")






# setup the prior
from bilby.core.prior import Uniform, PowerLaw, Sine, Constraint, Cosine
from corner import corner
import pandas as pd

# typically we would use a priors with wide bounds:
tc = trigger_time
priors = BBHPriorDict(dict(
    mass_ratio=Uniform(name='mass_ratio', minimum=0.125, maximum=1),
    chirp_mass=Uniform(name='chirp_mass', minimum=25, maximum=31),
    mass_1=Constraint(name='mass_1', minimum=10, maximum=80),
    mass_2=Constraint(name='mass_2', minimum=10, maximum=80),
    a_1=Uniform(name='a_1', minimum=0, maximum=0.99),
    a_2=Uniform(name='a_2', minimum=0, maximum=0.99),
    tilt_1=Sine(name='tilt_1'),
    tilt_2=Sine(name='tilt_2'),
    phi_12=Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic'),
    phi_jl=Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic'),
    luminosity_distance=PowerLaw(alpha=2, name='luminosity_distance', minimum=50, maximum=2000),
    dec=Cosine(name='dec'),
    ra=Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'),
    theta_jn=Sine(name='theta_jn'),
    psi=Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'),
    phase=Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic'),
    geocent_time=Uniform(minimum=tc - 0.1, maximum=tc + 0.1, latex_label="$t_c$", unit="$s$")
))

# however, for this example (to make analysis faster) we will use a prior with tighter bounds
priors['luminosity_distance'] = 419.18
priors['mass_1'] = Constraint(name='mass_1', minimum=30, maximum=50)
priors['mass_2'] = Constraint(name='mass_2', minimum=20, maximum=40)
priors['ra'] = 2.269
priors['dec'] = -1.223
priors['geocent_time'] = tc
priors['psi'] = 2.659
priors['theta_jn'] = 2.921
priors['phi_jl'] = 0.968


prior_samples = priors.sample(10000)
prior_samples_df = pd.DataFrame(prior_samples)
prior_samples_df


parameters = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2']

fig = corner(prior_samples_df[parameters], plot_datapoints=False, plot_contours=False, plot_density=True, color="tab:gray")
fig.show()

from bilby.gw.conversion import generate_mass_parameters

prior_samples = generate_mass_parameters(prior_samples)
prior_samples['cos_tilt_1'] = np.cos(prior_samples['tilt_1'])
prior_samples['cos_tilt_2'] = np.cos(prior_samples['tilt_2'])
s1z = prior_samples["a_1"] * prior_samples['cos_tilt_1']
s2z = prior_samples["a_2"] * prior_samples['cos_tilt_2']
q = prior_samples['mass_ratio']
prior_samples['chi_eff'] = (s1z + s2z * q) / (1 + q)
prior_samples_df = pd.DataFrame(prior_samples)
prior_samples_df


parameters = ['mass_1', 'mass_2', 'chi_eff']
fig = corner(prior_samples_df[parameters], plot_datapoints=False, plot_contours=False, plot_density=True, color="tab:gray")
fig.show()



# setup the waveform generator and likelihood



waveform_generator = WaveformGenerator(
    duration=interferometers.duration,
    sampling_frequency=interferometers.sampling_frequency,
    frequency_domain_source_model=lal_binary_black_hole,
    parameter_conversion=convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=20)
)

likelihood = GravitationalWaveTransient(
    interferometers=interferometers, waveform_generator=waveform_generator,
    priors=priors, time_marginalization=False, distance_marginalization=False,
    phase_marginalization=True, jitter_time=False
)


result = run_sampler(
    likelihood=likelihood, priors=priors, save=True,
    label="GW150914",
    nlive=50, walks=25,
    conversion_function=generate_all_bbh_parameters,
    result_class=CBCResult,
)


for interferometer in interferometers:
    fig = result.plot_interferometer_waveform_posterior(
        interferometer=interferometer
    )
    plt.show()


result.plot_corner(parameters=["mass_1_source", "mass_2_source", "chi_eff"])