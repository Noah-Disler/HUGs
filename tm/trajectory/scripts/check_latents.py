"""Validate the latents() hook against latents Dreamer stored during training.

The replay chunks contain `dyn/deter` and `dyn/stoch` recorded by policy() at
action-selection time. Re-encoding the same steps through the frozen checkpoint
should reproduce them closely.

Runs twice: once correctly aligned, once with actions deliberately shifted by
one step. The correct run must be clearly better, otherwise the comparison is
not sensitive enough to detect an alignment bug and cannot vouch for anything.
"""

# imports and paths
import sys
import pathlib
import numpy as np
import elements
import ruamel.yaml as yaml

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from embodied.jax import internal

LOGDIR = pathlib.Path('/datasets/ndisler/HUGs/logdir/20260903T154700')
WINDOW = 128

# load the config from the run, so the architecture matches the checkpoint
raw = elements.Path(str(LOGDIR / 'config.yaml')).read()
config = elements.Config(yaml.YAML(typ='safe').load(raw))
print('config loaded:', config.task, 'deter =', config.agent.dyn.rssm.deter)

# load one chunk and find a clean starting point. Starting at an episode
# boundary means both runs begin from a wiped RSSM state, so no warm start.
REPLAY = LOGDIR / 'replay'
names = sorted(p.name for p in REPLAY.glob('*.npz'))
print(f'found {len(names)} chunks')

for name in reversed(names):  # newest first: closest vintage to the checkpoint
  chunk = np.load(REPLAY / name)
  starts = np.flatnonzero(chunk['is_first'])
  if len(starts):
    print('using chunk:', name)
    break
else:
  raise SystemExit('no chunk contained an episode boundary')

i0 = int(starts[0])
# leave one step spare at the end so the shifted run can read action[i1]
i1 = min(i0 + WINDOW, len(chunk['is_first']) - 1)
print(f'episode starts at index {i0}, comparing {i1 - i0} steps')

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


def encode(shift):
  """Re-encode the slice. shift=0 is correct; shift=1 misaligns the actions.

  latents() computes prevact[t] = data['action'][t-1], so handing it actions
  that start one step later makes it use action[t] -- precisely the off-by-one
  we want to rule out, without touching the agent.
  """
  batch = {k: chunk[k][None, i0:i1] for k in keys}
  batch['action'] = chunk['action'][None, i0 + shift:i1 + shift]
  batch = internal.device_put(batch, agent.train_sharded)
  # same fixed seed for both runs, so sampling is not what separates them
  batch['seed'] = agent._seeds(0, agent.train_mirrored)
  carry = agent.init_report(1)
  _, outs = agent.latents(carry, batch)
  return outs


def report(label, outs):
  ours = np.asarray(outs['deter'])[0]
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
  z_ours = np.asarray(outs['stoch'])[0].argmax(-1)
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

# Reading the aligned run:
#   t=0 at the reset-spread noise floor  -> deter[0] is deterministic, so this
#                                           is the one exactly checkable value
#   error flat across t                  -> stable; a broken forward pass
#                                           compounds instead
#   error << step-to-step change         -> smaller than a real difference
# The model computes in bfloat16, so ~0.4% relative is the hard floor.
