import bilby
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(88170235)
OUTDIR = "outdir"

# Simulate signal
duration, sampling_freq, min_freq = 4, 1024., 20
injection_parameters = dict(
    mass_1=36.0, mass_2=29.0, # 2 mass parameters
    a_1=0.1, a_2=0.1, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, # 6 spin parameters
    ra=1.375, dec=-1.2108, luminosity_distance=2000.0, theta_jn=0.0, # 7 extrinsic parameters
    psi=2.659, phase=1.3,
    geocent_time=1126259642.413,
)
inj_m1, inj_m2 = injection_parameters['mass_1'], injection_parameters['mass_2']
inj_chirp_mass = bilby.gw.conversion.component_masses_to_chirp_mass(inj_m1, inj_m2 )
inj_q = bilby.gw.conversion.component_masses_to_mass_ratio(inj_m1, inj_m2)


waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_freq,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments= dict(
        waveform_approximant="IMRPhenomD",
        reference_frequency=20.0,
        minimum_frequency=min_freq,
    )
)

# fig = plot_waveform(waveform_generator, injection_parameters)




# Inject the signal into 1 detectors LIGO-Hanford (H1) at design sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_freq,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)


for interferometer in ifos:
    analysis_data = abs(interferometer.frequency_domain_strain)
    plt.loglog(interferometer.frequency_array, analysis_data, label="Data", color="tab:orange", alpha=0.25)
    plt.loglog(interferometer.frequency_array, abs(interferometer.amplitude_spectral_density_array),
               label="ASD (estimated noise)", color="tab:orange")
    plt.xlim(interferometer.minimum_frequency, interferometer.maximum_frequency)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r'Strain [strain/$\sqrt{\rm Hz}$]')
    plt.title(f"{interferometer.name} data")
    plt.legend()
    plt.show()


# Set up a PriorDict, which inherits from dict.
# By default we will sample all terms in the signal models.  However, this will
# take a long time for the calculation, so for this example we will set almost
# all of the priors to be equall to their injected values.  This implies the
# prior is a delta function at the true, injected value.  In reality, the
# sampler implementation is smart enough to not sample any parameter that has
# a delta-function prior.
# The above list does *not* include mass_1, mass_2, theta_jn and luminosity
# distance, which means those are the parameters that will be included in the
# sampler.  If we do nothing, then the default priors get used.
priors = bilby.gw.prior.BBHPriorDict()
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
    "theta_jn",
    "luminosity_distance",
]:
    priors[key] = injection_parameters[key]
priors["mass_ratio"] = inj_q
priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(minimum=inj_chirp_mass-5, maximum=inj_chirp_mass+5)


# Perform a check that the prior does not extend to a parameter space longer than the data
priors.validate_prior(duration, min_freq)

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator,
)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    npoints=250,
    nlive=100, walks=25,
    dlogz=0.1,
    injection_parameters=injection_parameters,
    outdir=OUTDIR,
    label="injection",
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    result_class=bilby.gw.result.CBCResult,
)


result.plot_corner(parameters=["mass_1", "mass_2"], truths=[inj_m1, inj_m2])

result = bilby.gw.result.CBCResult.from_json(outdir=OUTDIR, label="injection")
for interferometer in ifos:
    fig = result.plot_interferometer_waveform_posterior(
        interferometer=interferometer, save=False
    )
    plt.show()