"""ARCHIVED (this session): the exploratory version of
neural_experiments.py's run_n_neurons_convergence, extracted verbatim
before it was simplified down to the two settled SNR DVs (see
docs/HISTORY.md's own "n_neurons SNR measure exploration" entry for the
full narrative of how this was arrived at, and CLAUDE.md's neural_giant2
section for where these two DVs actually get used).

This version tested the CONVERGENCE HYPOTHESIS itself (does the
network's own running value estimate settle to nearly the same level
across seeds by the time the oddball hits, given 3 pre-observations
clustered within +-1 unit of each other) and, once that was found NOT to
hold cleanly for value-based cross-seed measures, went on to explore
several CANDIDATE purely-neural noise measures computed from the raw
error-population spike arrays it saved to a TEMPORARY folder
(data/runs/neural_experiments/tmp_spike_arrays/) -- most notably a
within-trial Fano factor (tried, found NOT to track n_neurons the way
every decoded measure does -- a single-neuron statistic has no mechanism
to capture the population-averaging benefit that decoding gets from
combining many independent noisy units) before landing on split-half
population reliability as the measure that actually worked.

NOT standalone-runnable as archived -- depends on this module's own
_base_params/_require_activities/_toy_activity_key/_decoders_for_seed/
_simulate_full and the CLI plumbing (--n_neurons_pairs parsing, argparse
subparser) that lived alongside it in scripts/neural_experiments.py.
Kept here as a reference snapshot of the measures tried and their exact
window/timing definitions, not for direct reuse.

How to restore: copy this function body back into scripts/
neural_experiments.py, restore its CLI subparser (see git history around
this same commit for the exact --n_neurons_pairs/--cluster_center/
--cluster_spread/--oddball_deviation/--alpha_0/--lambda_/--n_seeds args),
and re-add `import nengo`-adjacent helpers this file's own module-level
imports provide.
"""


def run_n_neurons_convergence(args) -> None:
    """Quick diagnostic (NOT a figure panel yet, local/cheap): tests
    whether the network's own running VALUE estimate genuinely converges
    to nearly the same level across seeds by the time the oddball hits,
    for the SAME oddball trial structure `oddball` already uses (3
    observations clustered around --cluster_center, then one deviating
    by --oddball_deviation), across --n_neurons_pairs. If convergence
    holds, cross-seed variance in PE/response AT the oddball should
    directly reflect momentary error-population SNR rather than
    accumulated drift from the 3 preceding observations.

    Keeps every seed's own RAW trace (does NOT pre-average across seeds
    the way `_oddball_worker` does), since testing convergence needs
    BOTH: (a) within-seed, across-TIME variance (momentary noise, for
    windows where a trace genuinely varies in time) and (b) across-seed
    variance of each seed's own window mean (drift/convergence). See
    this session's own chat for the exact reasoning; the companion
    report accompanying this function's own commit spells out exactly
    which windows/points feed which of the two.

    Measures the ERROR population (PE) ONLY during actual stimulus
    presentation (observation 1's own and the oddball's own), NOT during
    any ITI -- the error population is inhibited (~0 activity) during
    ITI, so PE variance there would just measure inhibition-floor
    artefacts, not SNR. Narrowed to a 200ms window centered on the
    established ~0.5s peak-response latency (t_iti+400ms to t_iti+600ms,
    relative to each observation's own onset), rather than the full
    stimulus presentation window, per instruction. The VALUE population,
    by contrast, is never inhibited and is measured across ITI windows
    too (this is in fact the DIRECT test of convergence itself: does the
    running estimate stay near the same level across seeds, or drift
    apart, during a period with no new input to explain any change).

    FINDING: convergence does NOT hold cleanly. value_iti_before_oddball's
    across-seed variance was 13-56x its own within-seed variance across
    all three n_neurons pairs tested -- most of the cross-seed spread in
    the running estimate by the time the oddball arrives is drift, not
    momentary noise, even with tightly clustered (+-1 unit) pre-
    observations. PE fared better but not perfectly (across/within ratio
    ~1.1-1.5x at observation 1, where no drift has had a chance to
    accumulate yet, vs ~2.2-3.5x at the oddball) -- still some drift
    contamination, just much less than the value-based measures.

    ALSO saves the RAW spike arrays for both 400-600ms windows (every
    seed, every neuron -- NOT restricted to weight-tuned, and NOT reduced
    to any summary statistic) to a clearly-marked TEMPORARY folder
    (data/runs/neural_experiments/tmp_spike_arrays/), per instruction --
    a Fano-factor-based purely-neural noise measure was tried and found
    NOT to track n_neurons the way every decoded measure does (Fano
    factor is a single-neuron statistic with no mechanism to capture the
    population-averaging benefit that decoding gets from combining many
    independent noisy units), so rather than committing to one
    alternative measure now, this saves the raw data itself so several
    candidate purely-neural measures (e.g. split-half population
    reliability, pairwise noise correlations) can be tried directly
    against it without rerunning any simulation.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"n_neurons_convergence_{args.task}.pkl"
    tmp_spike_dir = OUT_DIR / "tmp_spike_arrays"
    tmp_spike_dir.mkdir(parents=True, exist_ok=True)
    tmp_spike_path = tmp_spike_dir / f"spikes_{args.task}.pkl"

    cluster_vals = [args.cluster_center - args.cluster_spread, args.cluster_center,
                    args.cluster_center + args.cluster_spread]
    oddball_val = args.cluster_center + args.oddball_deviation
    obs_values_raw = np.array(cluster_vals + [oddball_val], dtype=float)
    obs_values = obs_values_raw / 50.0 - 1.0 if args.task == "soltani_numbers" else obs_values_raw

    pairs = []
    for spec in args.n_neurons_pairs:
        n_str, nc_str = spec.split(":")
        pairs.append((int(n_str), int(nc_str)))

    results = {}
    spike_data = {}
    dt_by_pair = {}
    for n_neurons, n_neurons_counting in pairs:
        params = _base_params(args.task, args.alpha_0, n_neurons, args.lambda_,
                              n_neurons_counting=n_neurons_counting)
        activity_map = _require_activities(args.task, n_neurons, n_neurons_counting)

        t_obs_ = float(params["t_obs"]); t_iti_ = float(params["t_iti"]); dt = float(params["dt"])
        t_step = t_obs_ + t_iti_
        dt_by_pair[(n_neurons, n_neurons_counting)] = dt

        per_seed = {
            "value_iti_obs1": [], "pe_obs1": [], "value_after_obs1": [],
            "value_iti_before_oddball": [], "pe_oddball": [], "value_decision": [],
        }
        spike_windows = {"obs1_spikes": [], "oddball_spikes": [], "encoders": []}
        for seed in range(args.n_seeds):
            key = _toy_activity_key(seed)
            decoders = _decoders_for_seed(activity_map, key, params["alpha_0"], params["lambda_"])
            result = _simulate_full(params, obs_values, decoders, seed=key)
            t_arr = result["t"]
            value_trace = result["value"]
            pe_trace = np.abs(result["pe_product"])

            m = (t_arr >= 0) & (t_arr < t_iti_)
            per_seed["value_iti_obs1"].append(value_trace[m])

            m = (t_arr >= t_iti_ + 0.4) & (t_arr < t_iti_ + 0.6)
            per_seed["pe_obs1"].append(pe_trace[m])
            spike_windows["obs1_spikes"].append(result["error_neurons"][m].copy())
            spike_windows["encoders"].append(result["encoders"].copy())

            m = np.abs(t_arr - t_step) < dt * 3
            per_seed["value_after_obs1"].append(float(np.mean(value_trace[m])))

            m = (t_arr >= 3 * t_step) & (t_arr < 3 * t_step + t_iti_)
            per_seed["value_iti_before_oddball"].append(value_trace[m])

            m = (t_arr >= 3 * t_step + t_iti_ + 0.4) & (t_arr < 3 * t_step + t_iti_ + 0.6)
            per_seed["pe_oddball"].append(pe_trace[m])
            spike_windows["oddball_spikes"].append(result["error_neurons"][m].copy())

            m = np.abs(t_arr - 4 * t_step) < dt * 3
            per_seed["value_decision"].append(float(np.mean(value_trace[m])))

        results[(n_neurons, n_neurons_counting)] = per_seed
        spike_data[(n_neurons, n_neurons_counting)] = spike_windows
        print(f"n_neurons_convergence: n_neurons={n_neurons} n_neurons_counting={n_neurons_counting} done")

    out = {
        "results": results,
        "obs_values_raw": obs_values_raw,
        "cluster_center": args.cluster_center,
        "cluster_spread": args.cluster_spread,
        "oddball_deviation": args.oddball_deviation,
        "alpha_0": args.alpha_0,
        "lambda_": args.lambda_,
        "n_seeds": args.n_seeds,
        "task": args.task,
    }
    pd.to_pickle(out, out_path)
    print(f"Saved -> {out_path}")

    spike_out = {
        "spike_data": spike_data,
        "dt": dt_by_pair,
        "n_seeds": args.n_seeds,
        "task": args.task,
    }
    pd.to_pickle(spike_out, tmp_spike_path)
    print(f"Saved raw spike arrays (TEMPORARY -- delete when done testing) -> {tmp_spike_path}")

    # Windowed measurements: within-seed variance (mean across seeds) AND
    # across-seed variance (of each seed's own window mean). Point
    # measurements (value_after_obs1, value_decision): across-seed
    # variance of the point value only -- "within-seed" doesn't apply to
    # a single small-window-averaged scalar.
    windowed = ["value_iti_obs1", "pe_obs1", "value_iti_before_oddball", "pe_oddball"]
    pointwise = ["value_after_obs1", "value_decision"]
    print(f"\n{'pair':<16} {'measure':<28} {'within-seed var (mean)':>24} {'across-seed var':>18}")
    for pair, per_seed in results.items():
        label = f"n={pair[0]},nc={pair[1]}"
        for measure in windowed:
            arrs = per_seed[measure]
            within = float(np.mean([np.var(a) for a in arrs]))
            means = [float(np.mean(a)) for a in arrs]
            across = float(np.var(means))
            print(f"{label:<16} {measure:<28} {within:>24.6f} {across:>18.6f}")
        for measure in pointwise:
            vals = per_seed[measure]
            across = float(np.var(vals))
            print(f"{label:<16} {measure:<28} {'n/a (point)':>24} {across:>18.6f}")


# ── Ad-hoc analysis snippets run directly against the saved spike arrays ────
# (data/runs/neural_experiments/tmp_spike_arrays/spikes_{task}.pkl) --
# NOT part of neural_experiments.py itself, run standalone against the
# pickle. Kept here for provenance since these are what actually
# determined the final split-half design (bin width, weight-tuned vs
# non-weight-tuned vs all neurons).
#
# Fano factor (tried, abandoned -- flat/noisy across n_neurons,
# mean~0.12-0.16, sd~0.03-0.07, no trend):
#
#   def _fano_factor_within_trial(spike_impulses, dt, bin_ms):
#       bin_size = int(round((bin_ms / 1000.0) / dt))
#       n_timesteps, n_neurons = spike_impulses.shape
#       n_bins = n_timesteps // bin_size
#       counts = spike_impulses[:n_bins*bin_size].reshape(
#           n_bins, bin_size, n_neurons).sum(axis=1) * dt
#       means = counts.mean(axis=0); variances = counts.var(axis=0)
#       valid = means > 0
#       return float(np.mean(variances[valid] / means[valid]))
#
# Split-half reliability (worked -- see docs/HISTORY.md for the full
# comparison across weight-tuned/all/non-weight-tuned subpopulations):
#
#   def bin_counts(spikes, dt, bin_ms):
#       bin_size = int(round((bin_ms/1000.0)/dt))
#       T, N = spikes.shape
#       n_bins = T // bin_size
#       return spikes[:n_bins*bin_size].reshape(n_bins, bin_size, N).sum(axis=1) * dt
#
#   def split_half_corr(counts, idx, n_splits, rng):
#       sub = counts[:, idx]; n = sub.shape[1]
#       corrs = []
#       for _ in range(n_splits):
#           perm = rng.permutation(n); half = n // 2
#           h1 = sub[:, perm[:half]].sum(axis=1)
#           h2 = sub[:, perm[half:2*half]].sum(axis=1)
#           if h1.std() > 0 and h2.std() > 0:
#               corrs.append(np.corrcoef(h1, h2)[0, 1])
#       return float(np.mean(corrs)) if corrs else np.nan
