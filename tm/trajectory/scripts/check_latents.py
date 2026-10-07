"""Validate the latents() hook against latents Dreamer stored during training.

The replay chunks contain `dyn/deter` and `dyn/stoch` recorded by policy() at
action-selection time. Re-encoding the same steps through the frozen checkpoint
should reproduce them closely.

Three checks:
  1. aligned -- re-encoding should track the stored latents closely
  2. misaligned control -- actions deliberately shifted by one step, which
     must be clearly worse, or the comparison is too insensitive to detect an
     alignment bug and cannot vouch for anything
  3. batch independence -- the same slice split across several forward passes
     with the carry threaded, as encode_states.py does. Error must not spike
     at the handoffs.

PIPELINE POSITION
  Validation only -- writes nothing. Run after any change to the latents()
  hook in dreamerv3/agent.py or embodied/jax/agent.py, before trusting
  encode_states.py output. Needs a GPU and the frozen checkpoint.
  Submit with cluster/check_latents.sbatch.

WHY IT EXISTS
  A one-step action misalignment is silent: shapes stay correct, nothing
  raises, and the latents look entirely plausible. The dyn/deter values that
  Dreamer recorded during training are the only ground truth available.

RESULT, 2026-09-30, frozen DoorKey-8x8 checkpoint, 56-step slice
  aligned     mean|err| 0.0154
  misaligned  mean|err| 0.0830 = 1.20x a real step-to-step change, i.e. worse
              than a genuine one-step difference, and it grows monotonically
  5.4x separation, so the test demonstrably detects what it guards against.
  t=0 agrees to 0.0020 in both runs -- the action is masked at a reset, so
  deter[0] is deterministic and this is the one exactly-checkable value. It
  sits at the 0.0025 spread between stored reset states, i.e. at the data's
  own noise floor. Residual error is sampling: stoch agreement 0.692.
  batch independence  0.85x at the carry handoff versus elsewhere, against a
  2.0 threshold and an expected ~6x if the carry were dropped.

  Caveat on that last one: SPLIT was 32 for this run and the chunk's longest
  episode was 56 steps, so only one handoff was tested. SPLIT is now 16,
  which gives three.
"""

# imports and paths
import sys
import pathlib
import numpy as np
import elements
import ruamel.yaml as yaml

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from embodied.jax import internal

import windows

LOGDIR = pathlib.Path('/datasets/ndisler/HUGs/logdir/20260903T154700')
WINDOW = 128
# Pass length for the batch-independence check. Small on purpose: late-training
# episodes run ~30-60 steps, so a larger value fits only one carry handoff in
# the window. 16 gives three.
SPLIT = 16

# load the config from the run, so the architecture matches the checkpoint
raw = elements.Path(str(LOGDIR / 'config.yaml')).read()
config = elements.Config(yaml.YAML(typ='safe').load(raw))
print('config loaded:', config.task, 'deter =', config.agent.dyn.rssm.deter)

# Load one chunk and pick the longest episode in it. Starting at an episode
# boundary means both runs begin from a wiped RSSM state, so no warm start is
# needed. Picking the *longest* episode also keeps resets out of the middle of
# the window, which matters for the batch-independence check below: a reset
# wipes the RSSM anyway, so it would mask a carry that was not being threaded.
REPLAY = LOGDIR / 'replay'
names = sorted(p.name for p in REPLAY.glob('*.npz'))
print(f'found {len(names)} chunks')

for name in reversed(names):  # newest first: closest vintage to the checkpoint
  chunk = np.load(REPLAY / name)
  episodes = windows.episode_bounds(chunk['is_first'], chunk['is_last'])
  if episodes:
    print('using chunk:', name)
    break
else:
  raise SystemExit('no chunk contained an episode boundary')

longest = max(episodes, key=lambda e: e.stop - e.start)
i0 = longest.start
# leave one step spare at the end so the shifted run can read action[i1]
i1 = min(i0 + WINDOW, longest.stop, len(chunk['is_first']) - 1)
resets_inside = int(chunk['is_first'][i0 + 1:i1].sum())
print(f'longest episode: {longest.stop - longest.start} steps at index {i0}')
print(f'comparing {i1 - i0} steps, {resets_inside} resets inside the window')

# Pre-flight, no GPU needed.
#
# Reference scales first. An absolute error is meaningless on its own: it needs
# a floor (bfloat16 gives ~0.4% relative precision, so nothing can beat that)
# and a ceiling (what unrelated vectors look like). Step-to-step change is the
# most useful of these -- it is the size of a difference that actually carries
# meaning, so an error well below it means we are tracking the real trajectory.
d = chunk['dyn/deter']
mag = np.abs(d).mean()
step = np.abs(np.diff(d, axis=0)).mean()
perm = np.random.default_rng(0).permutation(len(d))
unrelated = np.abs(d - d[perm]).mean()
print(f'deter magnitude:     {mag:.4f}')
print(f'step-to-step change: {step:.4f}  <- a real, meaningful difference')
print(f'unrelated vectors:   {unrelated:.4f}  <- what broken looks like')

# At a reset the RSSM zeroes deter, stoch and the action, so deter[0] =
# _core(0, 0, 0) -- a constant, for fixed weights. Measured spread here is the
# noise floor of the stored data, which t=0 below should match.
resets = d[chunk['is_first']]
if len(resets) > 2:
  print(f'reset spread over {len(resets)} starts: '
        f'adjacent={np.abs(np.diff(resets, axis=0)).mean():.4f}  '
        f'first-vs-last={np.abs(resets[0] - resets[-1]).mean():.4f}')

# build the agent and load the frozen weights
from dreamerv3 import main as dreamer_main

agent = dreamer_main.make_agent(config)

# ckpt/ is a container of versions: a `latest` pointer file naming the current
# one, beside the timestamped directory holding the actual weights. Training
# constructs Checkpoint(ckpt_dir) and resolves that itself, but the bare
# load() taken from eval_only.py needs a concrete path -- its exists() check
# looks for the `done` marker, which only the version directory has.
CKPT_DIR = LOGDIR / 'ckpt'
CKPT = CKPT_DIR / (CKPT_DIR / 'latest').read_text().strip()
print('loading checkpoint:', CKPT)

cp = elements.Checkpoint()
cp.agent = agent
cp.load(str(CKPT), keys=['agent'])

# latents() indexes every key in obs_space, so reward / is_last / is_terminal
# must be present even though the encoder ignores them.
keys = list(agent.obs_space) + list(agent.act_space)
theirs = chunk['dyn/deter'][i0:i1]


def encode(shift=0, pass_length=None):
  """Re-encode the slice, optionally split across several forward passes.

  shift=0 is correct; shift=1 misaligns the actions. latents() computes
  prevact[t] = data['action'][t-1], so handing it actions that start one step
  later makes it use action[t] -- precisely the off-by-one we want to rule
  out, without touching the agent.

  pass_length=None does the whole slice in one call. A smaller value threads
  the carry between calls exactly as encode_states.py does, which is what the
  batch-independence check below exercises.
  """
  total = i1 - i0
  pass_length = total if pass_length is None else pass_length
  carry = agent.init_report(1)
  deters, stochs = [], []
  for p, begin in enumerate(range(0, total, pass_length)):
    end = min(begin + pass_length, total)
    batch = {k: chunk[k][None, i0 + begin:i0 + end] for k in keys}
    batch['action'] = chunk['action'][
        None, i0 + begin + shift:i0 + end + shift]
    batch = internal.device_put(batch, agent.train_sharded)
    # Seeds restart at 0 for every encode() call, so the single-pass runs
    # above are seeded identically and sampling is not what separates them.
    batch['seed'] = agent._seeds(p, agent.train_mirrored)
    carry, outs = agent.latents(carry, batch)
    deters.append(np.asarray(outs['deter'])[0])
    stochs.append(np.asarray(outs['stoch'])[0])
  return {'deter': np.concatenate(deters, 0),
          'stoch': np.concatenate(stochs, 0)}


def report(label, outs):
  ours = outs['deter']
  assert ours.shape == theirs.shape, (ours.shape, theirs.shape)
  err = np.abs(ours - theirs)
  print(f'\n--- {label} ---')
  print(f'steps={len(ours)}  mean|err|={err.mean():.5f}')
  print(f'  / deter magnitude:     {err.mean() / mag:.4f}   '
        f'(bfloat16 floor is ~0.004)')
  print(f'  / step-to-step change: {err.mean() / step:.3f}   '
        f'(<<1 means we track the real trajectory)')
  print(f'  / unrelated vectors:   {err.mean() / unrelated:.3f}   '
        f'(near 1 means broken)')
  for t in [0, 1, 2, 5, 10, 20, 50, 100]:
    if t < len(ours):
      print(f'  t={t:3d}  mean|err|={err[t].mean():.5f}')
  z_ours = outs['stoch'].argmax(-1)
  z_theirs = chunk['dyn/stoch'][i0:i1].argmax(-1)
  print(f'  stoch sample agreement: {(z_ours == z_theirs).mean():.3f}')
  return err.mean()


good = report('aligned (prevact[t] = action[t-1])', encode(0))
bad = report('CONTROL: misaligned by one step', encode(1))

print(f'\nmisaligned / aligned = {bad / good:.2f}x')
if bad > good * 1.3:
  print('PASS: misalignment is clearly worse, so this test can detect it '
        'and the aligned run is trustworthy.')
else:
  print('INCONCLUSIVE: misalignment barely hurts, so this comparison cannot '
        'detect an alignment bug. Do not trust the aligned numbers as proof. '
        'Next step: seed the carry from the stored deter/stoch to remove the '
        'sampling confound and make the comparison deterministic.')

# --- batch independence -----------------------------------------------------
#
# encode_states.py threads the carry across ~121 forward passes per chain. If
# that were broken the RSSM would silently restart from zero at every pass
# boundary, putting a discontinuity every BATCH_STEPS into the states the
# autoencoder learns from, with nothing to signal it.
#
# Exact equality between one pass and several is not achievable: nj.scan
# derives per-step randomness from the pass, so a different pass length draws
# different z samples even with identical seeds. Instead compare both against
# the stored dyn/deter and look at *where* the error lands. A dropped carry
# makes deter at a boundary equal the reset constant _core(0, 0, 0), which is
# wrong by roughly the misaligned-run margin -- far above sampling noise.
split = encode(0, pass_length=SPLIT)
err_split = np.abs(split['deter'] - theirs)

boundaries = np.zeros(len(err_split), bool)
boundaries[np.arange(SPLIT, len(err_split), SPLIT)] = True
at_boundary = err_split[boundaries].mean()
elsewhere = err_split[~boundaries].mean()

print(f'\n--- batch independence ({len(err_split)} steps in passes of '
      f'{SPLIT}) ---')
print(f'single pass  mean|err| = {good:.5f}')
print(f'split passes mean|err| = {err_split.mean():.5f}')
print(f'  at the {int(boundaries.sum())} carry handoffs: {at_boundary:.5f}')
print(f'  everywhere else:              {elsewhere:.5f}')
print(f'  ratio: {at_boundary / elsewhere:.2f}x')

if at_boundary < elsewhere * 2:
  print('PASS: no error spike at pass boundaries, so the carry threads '
        'correctly and encode_states.py output is continuous.')
else:
  print('FAIL: error spikes where one pass hands over to the next, so the '
        'carry is not being threaded. encode_states.py output has a '
        'discontinuity every BATCH_STEPS and must be regenerated.')

# Reading the aligned run:
#   t=0 at the reset-spread noise floor  -> deter[0] is deterministic, so this
#                                           is the one exactly checkable value
#   error flat across t                  -> stable; a broken forward pass
#                                           compounds instead
#   error << step-to-step change         -> smaller than a real difference
# The model computes in bfloat16, so ~0.4% relative is the hard floor.
