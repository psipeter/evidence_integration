# neural-simulation-pipeline

Use this skill when generating extra NEF simulation data needed by figure
scripts — counting activity files, or data for the neural predictions
figure (`neural_main`) — as opposed to a standard model fit (see the
fitting-pipeline skill for that). Always generate locally (or via cluster
if slow), then scp to the cluster. NEF simulation runtime varies from
minutes to hours — write the script, then give the person the exact
command to run themselves once they've judged expected runtime; never
run one directly.

---

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

Older per-task carrabin/yoo neural panel data (PE dynamics, probe sims,
n_neurons scan, λ=0 ablation, error ensemble activities) is generated via
`scripts/extras_carrabin.py`/`scripts/extras_yoo.py` and
`jobs/submit_probe_pids.sh` — see those scripts' own `--help` for current
flags; this is the older N1-N8 taxonomy, not the current consolidated
argument.
