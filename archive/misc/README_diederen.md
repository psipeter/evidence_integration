# Diederen dataset

Archived from the active workflow.

Originally:
- Built from raw MATLAB files with:

```bash
python scripts/build_diederen.py --data_dir data/Diederen
```

- Groups: `CTRL` (n=28), `PCB` / `SUL` / `BRO` (n=19 each)
- Included model families: `Mean`, `RL`, `RL_lambda`, `PearceHall`, `NEF2d`
- Analysis and plotting scripts: `figure_diederen.py`, `dynamics_NEF2d.py`, `extras_diederen.py`

Raw data remains at `data/diederen.pkl`.
