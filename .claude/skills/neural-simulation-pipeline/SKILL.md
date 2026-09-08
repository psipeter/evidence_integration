# neural-simulation-pipeline

Use this skill when generating extra NEF simulation data needed by figure
scripts — counting activity files, or data for the neural predictions
figure (`neural_main`) — as opposed to a standard model fit (see the
fitting-pipeline skill for that). Also the reference for sizing/timeout
conventions when delegating a debug-sized NEF check to the
`nef-debug-runner` subagent (see below) — as opposed to a real
simulation, which still goes to the person. Always generate locally (or
via cluster if slow), then scp to the cluster. NEF simulation runtime
varies from minutes to hours — for a real run, write the script, then
give the person the exact command to run themselves once they've judged
expected runtime; never run a real one directly.

---

## Quick debug/sanity checks (`nef-debug-runner`)

For "does this new/changed NEF code even run, does it produce plausible
output" checks — not a real fit, not a result worth interpreting on its
own — delegate to the `nef-debug-runner` subagent
(`.claude/agents/nef-debug-runner.md`) rather than running it directly
in the main thread or skipping the check entirely. Its hard timeout
substitutes for the person's own runtime judgment, which only works
because the invocation is deliberately tiny. Confirmed working in
practice, not just a theoretical exception to the policy above.

Conventions for a debug-sized invocation:
- Use the SMALLEST already-precomputed counting-activity file available
  (`ls data/counting_activities_*.pkl` first) rather than generating a
  new one — generating one is itself a real simulation, out of scope
  for this agent.
- `--n_trials` well below the usual default (2-3, not 5+).
- A real, valid pid already present in the dataset's own pkl (check
  `data/{dataset}.pkl`'s `pid` column, or an existing `data/runs/*/`
  folder) — never a made-up pid.
- Timeout: 60-120 seconds is normally enough headroom at this size; if
  it times out, that's the signal to go smaller or hand it to the
  person — not to raise the timeout and retry.

Example (verified working):
```bash
venv/bin/python scripts/check_NEF_pipeline.py --dataset soltani_numbers \
    --pid 1 --alpha_0 0.3 --lambda_ 0.5 --n_neurons 50 \
    --n_neurons_counting 200 --n_trials 2
```

Still hand off to the person, per the policy above: any real fit,
anything meant to produce a trusted or publishable result, anything
needing the cluster, or anything that requires generating a missing
counting-activity file first.

## Counting activity files (required before NEF fitting)

```bash
venv/bin/python models/counting_integrator.py --precompute_activities \
    --n_neurons 200 --n_neurons_counting 1000 --dataset yoo --n_trials 30
scp data/counting_activities_n200_nc1000_yoo.pkl \
    f007qzn@discovery.dartmouth.edu:~/evidence_integration/data/
```

## Neural predictions figure (`neural_main`)

See `docs/SCIENCE.md`'s "Current thread" for what `neural_main` is and
its current status. Data source: `scripts/neural_experiments.py`'s
`oddball` (row 1, α₀) and `param_scan` (rows 2-3, λ/n_neurons)
experiments, run on `soltani_numbers`. Build the figure with:

```bash
python scripts/make_paper_figures.py neural_main
```

The older per-task carrabin/yoo neural panel data (PE dynamics, probe sims,
n_neurons scan, λ=0 ablation, error ensemble activities) and the N1-N8
taxonomy it fed (`figure_carrabin_neural.py`/`figure_yoo_neural.py`) are
retired — superseded by `neural_main`'s consolidated, soltani-only design
(see `docs/DECISIONS.md`). That older generation's scripts
(`extras_carrabin.py`, `extras_yoo.py`) and the job scripts that invoked
them are archived under `archive/scripts/`/`archive/jobs/`, restorable if
ever needed again.
