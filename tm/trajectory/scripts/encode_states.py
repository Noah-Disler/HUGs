"""Re-encode the saved replay through the frozen world model.

The autoencoder trains on model states s_t = concat(deter, flat(stoch)). Those
exist nowhere: the replay holds pixels and actions, and the dyn/deter stored
beside them was computed by an encoder that was still changing, so it is mixed
vintage. This pass runs every stored step through one frozen checkpoint and
writes a single consistent representation.

Run once. Everything downstream reads its output and never touches the replay.

Layout, one directory per chain:

    states/chain00/state.npy   [N, 640]    float32, memory-mappable
    states/chain00/logit.npy   [N, 32, 4]  float32
    states/chain00/index.npz   is_first, reward, and the episode table
    states/meta.json           checkpoint, seed, config -- provenance

deter and stoch are not stored separately: they are state[:, :512] and
state[:, 512:], so saving them would triple the size and invite the two copies
to drift apart. logit is kept because it is the one thing state cannot
reconstruct, and the sample-vs-probabilities ablation needs it.

No split is baked in. Chain id, episode index, return and completeness are all
recorded so a held-out-chains split, a random-episode split, or a
successful-only filter can each be made at load time.

PIPELINE POSITION
  replay chunks -> [this script] -> <logdir>/states/ -> dataset.py -> the
  autoencoder. Needs a GPU and the frozen checkpoint. Depends on the
  latents() hook in dreamerv3/agent.py and embodied/jax/agent.py, which
  scripts/check_latents.py verifies. Submit with cluster/encode_states.sbatch.

RESULT, 2026-09-30
  495,488 steps across 16 chains in 80s, 1.5 GB out. Episode totals match
  scripts/episode_stats.py exactly (4,832 episodes, 4,237 successful), which
  is a real cross-check: the two programs rebuild the chains independently.
"""

import itertools
import json
import pathlib
import sys
import time

import elements
import numpy as np
import ruamel.yaml as yaml

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from embodied.jax import internal  # noqa: E402

import windows  # noqa: E402

LOGDIR = pathlib.Path('/datasets/ndisler/HUGs/logdir/20260903T154700')
OUTDIR = LOGDIR / 'states'
# Steps per forward pass. Throughput only -- the carry makes the result
# identical whatever this is, and it has nothing to do with the window length
# the autoencoder will later use.
BATCH_STEPS = 256

raw = elements.Path(str(LOGDIR / 'config.yaml')).read()
config = elements.Config(yaml.YAML(typ='safe').load(raw))
print('config:', config.task, 'deter =', config.agent.dyn.rssm.deter)

REPLAY = LOGDIR / 'replay'
chains = windows.build_chains(windows.list_chunks(REPLAY))
print(f'chains: {len(chains)}  chunks: {sum(len(c) for c in chains)}')

from dreamerv3 import main as dreamer_main  # noqa: E402

agent = dreamer_main.make_agent(config)

CKPT_DIR = LOGDIR / 'ckpt'
CKPT = CKPT_DIR / (CKPT_DIR / 'latest').read_text().strip()
print('checkpoint:', CKPT)
cp = elements.Checkpoint()
cp.agent = agent
cp.load(str(CKPT), keys=['agent'])

KEYS = list(agent.obs_space) + list(agent.act_space)
# One seed per forward pass, advancing like stream() does. Deterministic
# across runs because the sequence starts from a fixed point.
seeds = itertools.count()


def encode_chain(data, length):
  """Encode one chain end to end, threading the carry between passes.

  Episode boundaries need no special handling: the RSSM zeroes deter, stoch
  and the previous action wherever is_first is set, so a chain can be fed as
  one continuous stream and each episode still starts from a wiped state.
  """
  carry = agent.init_report(1)
  states, logits = [], []
  for begin in range(0, length, BATCH_STEPS):
    end = min(begin + BATCH_STEPS, length)
    batch = {k: data[k][None, begin:end] for k in KEYS}
    batch = internal.device_put(batch, agent.train_sharded)
    batch['seed'] = agent._seeds(next(seeds), agent.train_mirrored)
    carry, outs = agent.latents(carry, batch)
    states.append(np.asarray(outs['state'])[0])
    logits.append(np.asarray(outs['logit'])[0])
  return np.concatenate(states, 0), np.concatenate(logits, 0)


OUTDIR.mkdir(parents=True, exist_ok=True)
start = time.time()
totals = {'steps': 0, 'episodes': 0, 'successful': 0}

for i, chain in enumerate(chains):
  data = windows.load_chain(REPLAY, chain, KEYS)
  length = len(data['is_first'])
  state, logit = encode_chain(data, length)
  assert len(state) == length, (len(state), length)

  episodes = windows.episode_bounds(data['is_first'], data['is_last'])
  returns = windows.episode_returns(data['reward'], episodes)

  out = OUTDIR / f'chain{i:02d}'
  out.mkdir(exist_ok=True)
  # Plain .npy rather than npz members, so the autoencoder's loader can
  # memory-map these instead of reading 1.5 GB into RAM.
  np.save(out / 'state.npy', state.astype(np.float32))
  np.save(out / 'logit.npy', logit.astype(np.float32))
  np.savez(
      out / 'index.npz',
      is_first=data['is_first'],
      reward=data['reward'],
      episode_start=np.array([e.start for e in episodes], np.int32),
      episode_stop=np.array([e.stop for e in episodes], np.int32),
      episode_complete=np.array([e.complete for e in episodes], bool),
      episode_return=returns.astype(np.float32),
  )

  totals['steps'] += length
  totals['episodes'] += len(episodes)
  totals['successful'] += int((returns > 0).sum())
  print(f'  chain {i:2d}: {length:6d} steps  {len(episodes):4d} episodes  '
        f'{int((returns > 0).sum()):4d} successful  '
        f'state={state.shape} {state.dtype}')

elapsed = time.time() - start
(OUTDIR / 'meta.json').write_text(json.dumps({
    'checkpoint': str(CKPT),
    'logdir': str(LOGDIR),
    'task': config.task,
    'deter': int(config.agent.dyn.rssm.deter),
    'stoch': int(config.agent.dyn.rssm.stoch),
    'classes': int(config.agent.dyn.rssm.classes),
    'state_dim': 640,
    'chains': len(chains),
    'batch_steps': BATCH_STEPS,
    'seed': int(config.seed),
    'dtype': 'float32',
    'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
    **totals,
}, indent=2))

print(f'\nwrote {len(chains)} chains to {OUTDIR}')
print(f"{totals['steps']} steps, {totals['episodes']} episodes, "
      f"{totals['successful']} successful, in {elapsed:.0f}s")
