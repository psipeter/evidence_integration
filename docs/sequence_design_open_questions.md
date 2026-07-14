# Sequence design: open questions (i.i.d. vs. quota/moment-matching)

**Status as of this writing: unresolved, real trade-off, not yet decided by the
PI.** Production currently uses the quota/moment-matched branch
(`generate_sequences_momentmatch.py`) -- the promoted 10x4 sequences already
live at `task/sequences/{task}_sequences.{pkl,json}` and are ready to ship.
Nothing in this memo blocks that from going out. This document exists so a
future session (or the PI) doesn't have to re-derive any of the following
from scratch, and so nobody accidentally "fixes" something here that was a
deliberate, considered trade-off rather than an oversight.

Read CLAUDE.md's "Sequence design" and "Sequence generation methods"
sections first for the mechanics (prefix/target independence, the collision
bug, optimal matching, etc.) -- this memo assumes that context and focuses
on the *behavioral/methodological* questions layered on top of it.

---

## 1. The triggering observation

A real pilot participant reported recognizing, partway through a sequence,
approximately where the true mean/probability was -- and using that to
discount later observations as "outliers" rather than genuinely updating on
them. This prompted a full investigation into whether the quota/moment-match
construction method makes this a *rational* strategy (not a bias) for
participants to discover, and whether that would bias derived behavioral
metrics (particularly the power-law decay rate lambda this project's whole
model-comparison pipeline is built around).

## 2. Is the quota confound real? Yes -- but not for the reason first proposed

**What turned out to be wrong**: the first framing was "seeing a streak
should make you predict a reversal" (classic gambler's-fallacy-is-correct
logic). This is only valid if the participant *knows* the quota/true
parameter in advance. They don't -- it's exactly what they're estimating --
so this specific directional claim doesn't hold up once corrected.

**What actually holds up, worked out rigorously (both analytically and via
direct simulation of the real generation code)**: the *variance* of the
running estimate, conditional on the true hidden parameter, shrinks faster
under quota-construction than under honest i.i.d. sampling -- provably so for
binary via the hypergeometric-vs-binomial finite-population-correction
factor `(N-n)/(N-1)`, and confirmed empirically for continuous via direct
simulation of the real `generate_sequences_momentmatch.py` code:

```
Continuous (target_mean=50, std=15): variance ratio (quota/iid) at each obs
  n=1: 0.96   n=5: 0.83   n=10: 0.38   n=14: 0.08   n=15: 0.00

Binary (target_p=0.6): variance ratio (quota/iid)
  n=1: 1.01   n=5: 1.56*  n=10: 0.44   n=14: 0.07   n=15: 0.00
  (*binary's prefix is INDEPENDENT of target, so early obs are noisier
  than i.i.d., not quieter -- see point 3 below)
```

The mechanism that survives careful scrutiny isn't "predict the opposite of
what you've seen" -- it's closer to **"whatever your running estimate is by
roughly the midpoint of a trial, trust it heavily, because the back half is
constructed to correct toward the true value almost regardless of how
wrong the front half was."** Confirmed directly:

```
Correlation between (how wrong you were at obs 10) and (how much the rest
of the trial corrects it):
  i.i.d.:   r=0.61  (loose -- no guaranteed correction)
  quota:    r=0.998 (near-deterministic -- correction is baked in)

Among the WORST 10% of early (obs-10) guesses:
  i.i.d.:   error 9.63 -> 6.37 by the end (partially rescued)
  quota:    error 5.85 -> 0.08 by the end (almost perfectly rescued)
```

This means terminal accuracy under quota is close to guaranteed by
construction, largely independent of how well a participant is actually
integrating evidence -- which undermines using terminal convergence (or
model fits that reward it) as a signal of genuine integration quality.

## 3. The prefix/suffix trilemma -- no single design satisfies all three goals

Three separate, real design goals turn out to be in tension, not just two:

1. **Statistical cleanliness** (no discontinuity between prefix and suffix --
   the whole 15-obs sequence looks like one honest distribution) -- needs the
   prefix to be tied to *a* target.
2. **Diversity** (many distinct targets, not capped at however many distinct
   prefixes exist) -- needs the prefix *not* tied to any one target.
3. **Repeat-based noise-isolation metrics (T5/T6, V-group)** -- these compute
   `residuals = response - mean(response | pid, obs, qid)`, which only
   cleanly isolates internal noise from stimulus-driven variance if repeats
   of a qid share **literal, identical stimulus history**, not just a
   shared abstract target. That literal repetition is also exactly what
   creates associative-memorization risk (recognizing "I've seen this exact
   opening before, I remember the answer").

The current momentmatch design (prefix independent of target) resolves (2)
and partially (3)'s memorization risk, at the cost of (1) -- this was a
deliberate, informed trade, not an oversight. Attempting to instead draw the
prefix from the *matched* target distribution (to fix (1)) was tried and
reverted: it reintroduces (2)'s diversity cap and creates a much sharper,
more literal (3) memorization risk (identical 4-observation openings,
repeated verbatim across a participant's own session).

**A real, previously-undetected bug found along the way**: this same
literal-repetition mechanism also affects `generate_sequences_iid.py`'s own
prefix generation (drawn from the matched target distribution, no
uniqueness check) -- empirically, **9 of 10 random seeds produced real
prefix collisions across different qids** (only 6-8 distinct prefixes out
of 10 expected). This was never checked before this investigation, since
all prior collision-bug fixing this project did was specific to
`generate_sequences_momentmatch.py`. Not yet fixed in `generate_sequences_iid.py`
-- flagged here so it's not forgotten if that branch is ever revisited.

**No design was found that fully satisfies all three goals simultaneously.**
This is presented as a real, acknowledged trade-off for the PI to weigh, not
a solved problem.

## 4. A model-recovery-based selection scheme was proposed and rejected

Idea considered: instead of quota-matching raw stimulus composition,
generate many i.i.d. sequence sets and keep whichever ones let reference
agents (Bayes/Mean, RL_lambda across various alpha/lambda) most accurately
recover their own known true parameters.

**Rejected** on the grounds that this is very likely the same underlying
mechanism as quota (outcome-conditioned selection from many i.i.d. draws),
just conditioned on a different property (parameter-recovery accuracy
instead of composition-near-target) -- CLAUDE.md's own earlier finding
already established that k-constrained rejection sampling and quota
sampling are "the same underlying object at different points on one
continuum"; this proposal doesn't escape that continuum, it just picks a
new point on it. It also introduces a *new*, more specific risk: sequences
selected because *these particular reference models* recover cleanly on
them would then be used to test whether real human behavior resembles
those same reference models -- a real circularity risk not present in
quota's model-agnostic conditioning on raw composition.

## 5. Ground truth choice (`true_mean`/`true_p` vs. `running_mean`) interacts
   with all of the above, and doesn't cleanly resolve it

`gt_mode='running_mean'` (in `scripts/inspect_sequences.py`) scores a
response against the running mean of *visible* data at that same moment --
never referencing the hidden true parameter at all. This structurally
sidesteps the quota confound (which is entirely about the relationship
between visible data and the hidden truth) for agents that are themselves
close to a raw running-mean tracker -- confirmed directly: this project's
own "Bayes"/"Mean" agent is *tautologically* perfect under `running_mean`
scoring (near-zero error always, both quota and i.i.d.), making that
gt_mode uninformative for that specific agent. For an agent with genuine
independent dynamics (RL_lambda), `running_mean` scoring not only removes
the quota advantage, it **reverses it** (quota scores slightly *worse* than
i.i.d. under this metric) -- confirmed empirically.

This is a real methodological fork with genuine precedent on both sides:
`true`-referenced scoring is standard in the Bayesian-updating/
volatility-learning literature (Behrens, Glaze, Nassar, Yu & Cohen), which
is explicitly interested in "how well do humans approximate the objectively
optimal observer of the real environment" -- unanswerable without
referencing the real parameter. `running_mean`-referenced scoring resonates
with "decisions from experience" literature, which argues a rational agent
should be judged against the actual finite sample it saw, not an
inaccessible population parameter. Neither is simply "more correct"; they
answer different questions. **Recommendation if this needs to go in a
paper**: report both, explicitly labeled by what each one tests, rather
than picking one silently.

## 6. Empirical results from `scripts/inspect_iid_sequences.py`

Built this session specifically to investigate all of the above with real
numbers rather than argument alone. Generates N independent i.i.d. sequence
sets (one fully independent draw per simulated participant, via the real
`generate_sequences_iid.py` code -- see that script's own docstring for
exactly what "independent" means here), simulates a chosen agent (Mean or
RL_lambda) on each, and reports fitted-lambda mean/std/range plus
split-half reliability across the simulated population.

**Headline results, n=50 simulated participants, Mean agent** (true
lambda=1 by construction -- Mean is RL_lambda's own special case
alpha_0=1, lambda=1):

```
                fitted lambda (mean+/-std)   range          split-half reliability
continuous      1.13 +/- 0.29                [0.60, 1.85]    r=0.82, p<0.0001
binary          0.77 +/- 0.17                [0.38, 1.28]    r=0.76, p<0.0001
```

Split-half reliability is strong for both tasks despite every participant
seeing completely independently-randomized sequences -- i.i.d. sampling
noise alone does *not* wash out the ability to detect a consistent decay
signature. But **the population mean itself sits meaningfully off the true
value of 1** (especially for binary), and the spread across participants is
substantial -- a real cost of not smoothing/selecting sequences at all.

**Comparison against the real (quota, seed-searched) production
sequences**, same Mean agent, single dataset (not averaged over many
draws):

```
                i.i.d. mean (n=50)    production (quota, n=1)
continuous      1.133                 1.044   (much closer to true=1)
binary          0.765                 0.758   (essentially unchanged)
```

Quota substantially helps continuous lambda-recovery, but **does nothing
measurable for binary**. Important caveat this comparison surfaced: this
conflates quota construction itself with the 1000-try seed search that
selected this specific sequence set for smoothness -- an *unselected*
single quota draw might look much more like the i.i.d. distribution. Not
yet checked; flagged as a natural follow-up if this matters for a real
decision.

**A second, independent bias was found and isolated**: binary's poor
lambda recovery (both i.i.d. and quota alike) is substantially explained by
the Laplace-smoothing transform (`utils/binary_transform.py`) applied
uniformly to every non-exempt model's binary output project-wide -- its
stated justification ("optimal Bayesian estimate under a uniform prior")
is specific to a raw sample-mean estimator and doesn't clearly generalize.
Confirmed by comparing fits with vs. without the transform:

```
                  WITH transform   WITHOUT transform   true lambda
Mean, binary      0.758            0.903                1.0
RL_lambda(1, 0.5) 0.284            0.430                0.5
RL_lambda(0.5,0.3) 0.158           0.323                0.3
```

Removing the transform closes 60-90% of the gap to the true value in every
case checked, at the cost of somewhat higher variance. **This bias is
orthogonal to the i.i.d.-vs-quota question** -- it affects both branches
equally, since it's applied after sequence generation, not as part of it.
`scripts/inspect_iid_sequences.py` now supports `--skip_binary_transform`
to isolate this. Not resolved/decided whether production analyses should
use the transform or not -- flagged, not fixed, since it affects more than
just this investigation (it's already used project-wide).

**RL_lambda noise sensitivity** (as predicted going in -- smaller lambda
means alpha decays slower, so individual noisy observations keep mattering
later into a trial): confirmed clearly noisier than the Mean agent,
especially for binary (coefficient of variation roughly doubles), and the
specific bias/noise trade-off shifts in a non-obvious way with alpha_0
(one config showed large systematic bias with *low* variance; another
showed smaller bias with *higher* variance) -- this isn't simply "smaller
lambda -> uniformly more noise," alpha_0 is doing real, not-yet-fully-
characterized work too.

## 7. What it would take to actually serve unique i.i.d. sequences per
   participant, if that path is ever chosen

Full breakdown was worked through in chat; summary:

- **Sequence delivery**: currently a build-time static import
  (`config.js` imports `{task}_sequences.json` directly, baked into every
  participant's identical bundle). Serving per-participant pools requires a
  real architecture change: bundle a pool of files as static assets, fetch
  the assigned one at runtime, and make the app's bootstrap sequence async
  on that fetch resolving. Needs live confirmation that JATOS actually
  serves additional static files from a study's asset directory via
  relative-path `fetch()` the way this assumes -- not yet verified against
  real JATOS.
- **Pool generation**: mostly already built -- `simulate_participants` in
  `inspect_iid_sequences.py` already generates N independent sets; turning
  that into a real pool-writer is a small wrapper, not new work.
- **Assignment**: recommend a stateless hash of participant ID mod pool
  size (e.g. Prolific PID) -- avoids needing any server-side coordination or
  counter, which would carry real concurrency risk (two participants
  starting simultaneously racing on a shared counter).
- **Recording what a participant saw -- better news than expected**: the
  raw per-observation JATOS export *already* includes `value` (and, from
  direct inspection of real pilot files, `true_mean`/`true_p`) -- jsPsych
  automatically records trial parameters. The actual gap is in
  `parse_results.py`, which currently *discards* these and instead
  re-derives `value` via a lookup against the single shared sequence file
  (documented explicitly in that script's own docstring), assuming
  `(task, trial)` uniquely determines it -- true only while everyone shares
  one file. The fix is smaller than it first appears: extract these fields
  directly from each participant's own raw row instead of re-deriving them,
  which is *also* a general robustness improvement (removes an existing,
  if currently harmless, fragility where a regenerated sequence file could
  silently produce mismatched historical lookups). Worth confirming exactly
  which mechanism currently puts `true_mean`/`true_p` onto observation rows
  (direct trial parameter vs. a retroactive `jsPsych.data.addProperties()`
  call, which has bitten this project before -- see `trial_timeouts`'
  history in CLAUDE.md) before relying on it. Recording a session-level
  `pool_index` property in addition is cheap and removes any ambiguity
  about provenance.
- **Cross-participant analysis**: a genuine methodology change, not just
  plumbing -- once trial index no longer means the same target for
  everyone, anything that aggregates "by trial number" needs to instead
  align by target value or by qid-within-that-participant's-own-pool-member.
- **Diagnostics**: `inspect_sequences.py` assumes one shared file; would
  need to inspect one representative pool member or aggregate across the
  whole pool (the aggregation machinery for the latter already exists in
  `inspect_iid_sequences.py`).
- **Ongoing costs worth naming**: total asset storage scales with pool
  size; debugging a specific participant's data now requires knowing which
  pool member they got; this becomes real standing infrastructure to
  maintain for the life of the study, not a one-time script.

## 8. Where this leaves things

Nothing here should be read as "quota is wrong, switch to i.i.d." or the
reverse. Every path has real, now-quantified costs:

- **Quota (current production)**: real behavioral confound established with
  numbers, not hand-waving (Section 2), plus a fundamental prefix-design
  trilemma with no fully clean resolution (Section 3).
- **Pure i.i.d.**: no behavioral confound from construction, but
  substantially noisier lambda-recovery even averaged over many
  independent draws (Section 6), a real (if fixable) collision bug in its
  own prefix generation (Section 3), and would require substantial new
  infrastructure to actually deploy with per-participant uniqueness
  (Section 7) -- or, if deployed with one shared i.i.d. draw (the simpler
  option), inherits the "got lucky/unlucky with the one seed" risk that
  per-participant randomization was meant to avoid.
- **Model-recovery-based selection**: rejected (Section 4) as likely just
  quota again, with an added circularity risk.

This is a real trade-off for the PI to make with full information, not a
default to quietly pick. The current 10x4 quota-based production sequences
are already generated, verified, and ready to ship regardless of how this
resolves -- nothing here is blocking deployment.
