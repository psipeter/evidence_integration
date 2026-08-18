"""utils/soltani_models.py

Single definition of WHICH models the three soltani figures know about, and which
of them each panel may plot. Shared so figure_soltani_{performance,temporal,
variability}.py cannot drift apart -- they previously each carried their own
MODEL_ORDER copy, which is the same failure mode that let the three temporal
figures end up with three different aggregation schemes (see utils/aggregate.py).

MODEL_ORDER is the full vocabulary AND the colour order: utils.plot_style.
get_palette returns the first n of a fixed colourblind list, so a model's colour
is its INDEX here. Consequences worth knowing before editing:
  - APPEND new models, never insert. Inserting shifts every later model's colour
    and silently makes new figures incomparable with old ones.
  - Palettes are always built over the FULL MODEL_ORDER, never over a requested
    subset, so a model keeps its colour in any subset (--models RL_lambda gives
    the same orange as the full figure).

DEFAULT_MODELS is deliberately just the three baselines. RL_lambda,
NoisyRL_lambda and NEF are opt-in via --models, so a default figure stays
readable and the richer models are shown when they are the point.

STOCHASTIC_MODELS is a separate axis from MODEL_ORDER: it is the subset whose
responses VARY across repeats of an identical stimulus prefix. Panels built on
within-qid residuals -- temporal cols 3-4, and all of the variability figure --
are meaningless for a deterministic model, whose residual against a
qid-conditional mean is EXACTLY zero (verified: max|resid| 0.000e+00 for Mean,
LeakyIntegrator, PrimacyRecency and RL_lambda over 1152 prefix rows, against 0.68
for Human). Those panels filter on this set, so requesting only deterministic
models leaves them human-only rather than drawing flat lines at zero.
"""
from __future__ import annotations

# Append-only; index determines colour. NoisyRL_lambda is last so that adding it
# left every existing model's colour untouched.
MODEL_ORDER = [
    "Mean",
    "LeakyIntegrator",
    "PrimacyRecency",
    "RL_lambda",
    "NEF",
    "NoisyRL_lambda",
]

DEFAULT_MODELS = ["Mean", "LeakyIntegrator", "PrimacyRecency"]

# Models with a genuine noise term. NoisyRL_lambda qualifies only when its noise
# is actually nonzero -- under RMSE fitting both sigmas pile up at their lower
# bounds (measured: 35/35 at floor for numbers), so with floors of 0 it would
# behave exactly like RL_lambda and draw a flat line here.
STOCHASTIC_MODELS = frozenset({"NEF", "NoisyCounting", "NoisyRL_lambda"})


def add_model_args(parser, default: list[str] | None = None) -> None:
    """Add --models to a soltani figure's parser, worded identically everywhere."""
    d = list(default if default is not None else DEFAULT_MODELS)
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        help="Which fitted models to overlay, in this order. Choose from "
             f"{', '.join(MODEL_ORDER)}. Default {', '.join(d)} -- the three "
             "baselines; RL_lambda, NoisyRL_lambda and NEF are opt-in so a "
             "default figure stays readable. Pass --models with no effect on "
             "colours: each model's colour comes from its position in the full "
             "MODEL_ORDER, not from the requested subset, so subset figures stay "
             "comparable with full ones. Panels built on within-qid residuals "
             "additionally keep only the stochastic models "
             f"({', '.join(sorted(STOCHASTIC_MODELS))}), since a deterministic "
             "model's residual against a qid-conditional mean is exactly zero.")


def resolve_models(requested: list[str] | None, parser=None,
                   default: list[str] | None = None) -> list[str]:
    """Validate a --models list, or return the default. Order is preserved."""
    if not requested:
        return list(default if default is not None else DEFAULT_MODELS)
    unknown = [m for m in requested if m not in MODEL_ORDER]
    if unknown:
        msg = f"unknown model(s) {unknown}; choose from {MODEL_ORDER}"
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)
    return list(requested)


def stochastic_only(models: list[str]) -> list[str]:
    """Subset of `models` eligible for within-qid-residual panels."""
    return [m for m in models if m in STOCHASTIC_MODELS]
