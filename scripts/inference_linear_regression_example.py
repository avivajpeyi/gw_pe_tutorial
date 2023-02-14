"""
An example of how to use bilby to perform parameter estimation for
fitting a linear function to data with background Gaussian noise.
This will compare the output of using a stochastic sampling method
to evaluating the posterior on a grid.
"""
import bilby
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from corner.core import quantile

np.random.seed(42)

# A few simple setup steps
label = "linear_regression_grid"
outdir = "."


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict()
injection_parameters["c"] = 0.2
injection_parameters["m"] = 0.5

# For this example, we'll use standard Gaussian noise

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
sigma = 3.0
data = model(time, **injection_parameters) + np.random.normal(0, sigma, N)

# We quickly plot the data to check it looks sensible
fig, ax = plt.subplots()
ax.plot(time, data, "o", label="data", color='k')
ax.plot(time, model(time, **injection_parameters), "--r", label="signal")
ax.set_xlabel("time")
ax.set_ylabel("y")
ax.legend()
fig.savefig("{}/{}_data.png".format(outdir, label))

# Now lets instantiate a version of our GaussianLikelihood, giving it
# the time, data and signal model
likelihood = bilby.likelihood.GaussianLikelihood(time, data, model, sigma)


# We make a prior


def get_m_c_prior():
    priors = bilby.core.prior.PriorDict(dict(
        m=bilby.core.prior.Uniform(0, 4, "m"),
        c=bilby.core.prior.Uniform(-2, 2, "c"),
    ))
    return priors


m_c_prior = get_m_c_prior()


def get_grid_of_m_c(n_samples):
    prior = get_m_c_prior()
    n_per_dim = int(np.ceil(n_samples ** (1 / len(prior))))
    m_vals = np.linspace(prior['m'].minimum, prior['m'].maximum, n_per_dim)
    c_vals = np.linspace(prior['c'].minimum, prior['c'].maximum, n_per_dim)
    m_grid, c_grid = np.meshgrid(m_vals, c_vals)
    return pd.DataFrame(dict(m=m_grid.flatten(), c=c_grid.flatten()))


def brute_force_run(likelihood, prior, n_samples):
    grid = get_grid_of_m_c(n_samples=n_samples)
    n_samples = len(grid)
    log_likelihoods = np.zeros(n_samples)
    log_priors = np.zeros(n_samples)
    for i in tqdm(range(n_samples)):
        sample = grid.iloc[i].to_dict()
        likelihood.parameters = sample
        log_likelihoods[i] = likelihood.log_likelihood()
        log_priors[i] = prior.ln_prob(sample)
    log_evidence = np.logaddexp.reduce(log_likelihoods + log_priors)
    grid["log_posterior"] = (log_likelihoods + log_priors) - log_evidence
    grid["log_prior"] = log_priors
    grid["log_likelihood"] = log_likelihoods
    print("Brute force ln_evidence: {}".format(log_evidence))
    return grid, log_evidence


def get_marginalised_posterior(parameter, grid):
    unique_values = np.unique(grid[parameter])
    log_posterior = np.zeros(len(unique_values))
    for i, value in enumerate(unique_values):
        log_posterior[i] = np.logaddexp.reduce(grid[grid[parameter] == value]["log_posterior"])
    return unique_values, np.exp(log_posterior)


def sampler_run():
    # And run sampler
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=get_m_c_prior(),
        sampler="dynesty",
        nlive=500,
        sample="unif",
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
    )
    return result


def plot_grid(grid, injection_parameters=None, axes=None, save=False):
    m_vals, p_m = get_marginalised_posterior("m", grid)
    c_vals, p_c = get_marginalised_posterior("c", grid)
    n_cells = int(np.sqrt(len(grid)))
    m_grid = grid['m'].values.reshape(n_cells, n_cells)
    c_grid = grid['c'].values.reshape(n_cells, n_cells)
    posterior_grid = np.exp(grid['log_posterior'].values.reshape(n_cells, n_cells))
    posterior_grid = posterior_grid / np.sum(posterior_grid)

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(5,5))
        axes = axes.flatten()
        axes[1].axis("off")

    fig = axes[0].figure
    c_axis = axes[0]
    m_c_axis = axes[2]
    m_axis = axes[3]

    m_axis.plot(m_vals, p_m, color="tab:green", alpha=0.8)
    c_axis.plot(c_vals, p_c, color="tab:green", alpha=0.8)
    m_axis.set_ylim(0, 1.15 * np.max(p_m))
    c_axis.set_ylim(0, 1.15 * np.max(p_c))
    m_c_axis.contourf(c_grid, m_grid, posterior_grid, levels=np.quantile(posterior_grid, [0.95, 0.99, 1]), cmap="Greens")

    for axi in [c_axis, m_axis]:
        axi.set_yticks([])

    if injection_parameters is not None:
        m_axis.axvline(injection_parameters["m"], color="tab:orange", )
        c_axis.axvline(injection_parameters["c"], color="tab:orange", )
        m_c_axis.axhline(injection_parameters["m"], color="tab:orange", )
        m_c_axis.axvline(injection_parameters["c"], color="tab:orange", )
        m_c_axis.scatter(injection_parameters["c"], injection_parameters["m"], color="tab:orange", marker='s')

    if save:
        # zoom in on the region of interest
        m_quan = quantile(m_vals, [0.002, 0.998], weights=p_m)
        c_quan = quantile(c_vals, [0.002, 0.998], weights=p_c)
        m_axis.set_xlim(m_quan)
        c_axis.set_xlim(c_quan)
        m_c_axis.set_xlim(c_quan)
        m_c_axis.set_ylim(m_quan)
        c_axis.set_xticks([])
        m_axis.set_xlabel("m", fontsize=16)
        m_c_axis.set_ylabel("m", fontsize=16)
        m_c_axis.set_xlabel("c", fontsize=16)
        m_axis.set_title(f"p(m|d)")
        c_axis.set_title(f"p(c|d)")
        fig.tight_layout()
        fig.savefig("brute_force.png")


def overplot_sampler_and_brute_force(sampler_result, grid, injection_parameters):
    n_sampler = len(sampler_result.posterior)
    n_grid = len(np.unique(grid["m"]))
    weights = np.ones(n_sampler) * (n_grid / n_sampler)
    fig = sampler_result.plot_corner(parameters=injection_parameters, save=False, weights=weights)
    # overplot the grid estimates
    axes = fig.axes
    for axi in [0, 3]:
        axes[axi] = axes[axi].twinx()
        axes[axi].set_yticks([])
    grid_ln_evidence = np.logaddexp.reduce(grid["log_likelihood"] + grid["log_prior"])
    plot_grid(grid, axes=axes, save=False)
    fig.savefig("comparison.png", dpi=300)
    print("Brute force ln_evidence: {}".format(grid_ln_evidence))
    print("Sampler ln_evidence: {}".format(sampler_result.log_evidence))


def plot_posterior_predictive_check(
    observed_data,
    time,
    model,
    posterior,
):
    # take 100 random samples from the posterior
    post = posterior.sample(1000)
    m, c = post['m'].values, post['c'].values
    ys = np.array([model(time, mi, ci) for mi, ci in zip(m, c)]).T
    y_low, y_mean, y_up = np.quantile(ys, [0.05, 0.5, 0.95], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.scatter(time, observed_data, color="k", zorder=-10, s=1)
    ax.plot(time, y_mean, color="tab:blue")
    ax.fill_between(time, y_low, y_up, alpha=0.1, color="tab:blue")
    ax.set_xlim(time.min(), time.max())
    plt.title("Posterior predictive check")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig.savefig("posterior_predictive_check.png", dpi=300)


grid, log_evid = brute_force_run(likelihood, m_c_prior, 10000)
sampler_result = sampler_run()
overplot_sampler_and_brute_force(sampler_result, grid, injection_parameters)
plot_posterior_predictive_check(data, time, model, sampler_result.posterior)
plot_grid(grid, injection_parameters, save=True)

fig, ax = plt.subplots()

# remove axes splines
ax.