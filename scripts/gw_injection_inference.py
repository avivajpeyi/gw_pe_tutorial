import numpy as np
import bilby
from bilby import run_sampler
import matplotlib.pyplot as plt
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
import logging

bilby_logger = logging.getLogger("bilby")

# ## Downloading IFO data
#
# Now we download the raw data and make some plots

# +
interferometers = InterferometerList(["H1", "L1"])
tc = get_event_time("GW150914")

start_time = tc - 3
duration = 4
end_time = start_time + duration
roll_off = 0.2
maximum_frequency = 512
minimum_frequency = 20



# Get raw data
raw_data = {}
for ifo in interferometers:
    print(
        f"Getting {ifo.name} analysis data segment (takes ~ 30s)"
    )
    analysis_data = TimeSeries.fetch_open_data(
        ifo.name, start_time, end_time
    )
    ifo.strain_data.roll_off = roll_off
    ifo.strain_data.set_from_gwpy_timeseries(analysis_data)
    raw_data[ifo.name] = analysis_data
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency

# -

# plot raw data:
plot = GWpyPlot(figsize=(12, 4.8))
ax = plot.add_subplot(xscale='auto-gps')
for ifo_name, data in raw_data.items():
    ax.plot(data, label=ifo_name)
ax.set_epoch(tc)
ax.set_xlim(1126259462, 1126259462.6)
ax.set_ylabel('Strain noise')
ax.legend()
plot.show()

# Woah. That looks terrible. Where is the nobel-prize winning poster-child signal?
#
# We may need to clean up the data a bit to actually 'see' the signal. Lets get the data for the PSD and take a look at the noise once again

# +
# downloading data
psd_start_time = start_time + duration
psd_duration = 128
psd_end_time = psd_start_time + psd_duration
psd_tukey_alpha = 2 * roll_off / duration
overlap = duration / 2

for interferometer in interferometers:
    print(
        f"Getting {interferometer.name} PSD data segment (takes ~ 1min)"
    )
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

# -

# plotting
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

# Lets `notch` out the 60 and 120 Hz `violin` modes (black vertical lines), and only keep data within the 50-250Hz range (marked in green) from the raw data and re-plot:

plot = GWpyPlot(figsize=(12, 4.8))
ax = plot.add_subplot(xscale='auto-gps')
for ifo_name, data in raw_data.items():
    filtered_data = data.bandpass(50, 250).notch(60).notch(120)
    ax.plot(filtered_data, label=ifo_name)
ax.set_epoch(tc)
ax.set_xlim(1126259462, 1126259462.6)
ax.set_ylim(-1e-21, 1e-21)
ax.set_ylabel('Strain noise')
ax.legend()
plot.show()

# Noiceee... Lets plot the signals in the frequency domain:

# +


for ifo_name, data in raw_data.items():
    qtrans = data.q_transform(
        frange=(20, 500), fres=0.05,
        outseg=(tc - 0.2, tc + 0.1)
    )
    plot = qtrans.plot(cmap='viridis', dpi=150)
    ax = plot.gca()
    ax.set_title(f'{ifo_name} Q-transform')
    ax.set_epoch(tc)
    ax.set_yscale('log')
    ax.colorbar(label="Normalised energy")

# -

# ## Getting priors
#
# Now lets write down our priors for this event's analysis. Note that again we set several delta functions and restrict the search space to speed up analysis for the sake of this tutorial.

# +
# setup the prior
from bilby.core.prior import Uniform, PowerLaw, Sine, Constraint, Cosine
from corner import corner
import pandas as pd

# typically we would use a priors with wide bounds:
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
priors['theta_jn'] = 2.921
priors['phi_jl'] = 0.968
priors['psi'] = 2.659

prior_samples = priors.sample(10000)
prior_samples_df = pd.DataFrame(prior_samples)

# -

# Plots of some priors:
#

parameters = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2']
fig = corner(
    prior_samples_df[parameters], plot_datapoints=False,
    plot_contours=False, plot_density=True,
    color="tab:gray"
)

# Lets convert some parameters and see what the prior-distributions we get

# +
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
fig = corner(prior_samples_df[parameters], plot_datapoints=False, plot_contours=False, plot_density=True,
             color="tab:gray")

# -

# ## Inference step

# +
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

RE_RUN_SLOW_CELLS = True
if RE_RUN_SLOW_CELLS:
    bilby_logger.setLevel(logging.INFO)
    result = run_sampler(
        likelihood=likelihood, priors=priors, save=True,
        label="GW150914",
        nlive=1000,
        conversion_function=generate_all_bbh_parameters,
        result_class=CBCResult,
    )
else:
    print("Skipping sampling...")
    fn = f"{OUTDIR}/GW150914_result.json"
    download(GW150914_URL, fn)
    result = bilby.gw.result.CBCResult.from_json(filename=fn)
    print("Loaded result!")

# -

result.plot_corner(parameters=["mass_ratio", "chi_eff"])

result.plot_corner(parameters=["mass_1", "mass_2"])
