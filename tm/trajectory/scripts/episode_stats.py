"""Report the episode structure of the saved replay.

Window length cannot be chosen responsibly without this. A trained DoorKey-8x8
agent finishes in roughly 20-40 steps while early episodes run to the 500-step
cap, so a long window silently excludes every competent episode and leaves a
dataset made entirely of incompetent behaviour -- with no error to notice.

Reads only is_first / is_last / reward. npz decompresses lazily, so skipping
`image` makes this cost seconds rather than minutes.

PIPELINE POSITION
  Analysis only -- reads the replay, writes nothing. Run before choosing a
  window length, or after collecting new data. CPU only, no checkpoint.
  Submit with cluster/episode_stats.sbatch (stampede partition, no GPU).

RESULT, 2026-09-30, DoorKey-8x8 replay at logdir/20260903T154700
  16 chains x 32 chunks, 495,488 steps, 4,832 episodes, 87.7% successful.
  Successful episodes: p50=28, p90=92. Failures: 501 steps almost always,
  i.e. failure is timeout. W=16 retains 96% of successful episodes; W=32
  retains 40% and skews the surviving set toward failures.
  Note the separate problem this surfaced: failures are ~18x longer, so a
  step-uniform sample is ~64% failure behaviour even at W=16. Corrected by
  windows.balanced_weights, not by window length.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import windows  # noqa: E402

LOGDIR = pathlib.Path('/datasets/ndisler/HUGs/logdir/20260903T154700')
REPLAY = LOGDIR / 'replay'
PCTS = [0, 10, 25, 50, 75, 90, 100]

names = windows.list_chunks(REPLAY)
chains = windows.build_chains(names)
print(f'chunks: {len(names)}')
print(f'chains: {len(chains)}  (expect one per environment)')
print(f'chunks per chain: {sorted(len(c) for c in chains)}')

lengths, complete, success, position = [], [], [], []
total_steps = 0

for i, chain in enumerate(chains):
  data = windows.load_chain(REPLAY, chain, ['is_first', 'is_last', 'reward'])
  eps = windows.episode_bounds(data['is_first'], data['is_last'])
  total_steps += len(data['is_first'])
  for j, ep in enumerate(eps):
    lengths.append(ep.stop - ep.start)
    complete.append(ep.complete)
    success.append(float(data['reward'][ep.start:ep.stop].sum()) > 0)
    # where this episode sits in the chain, as a proxy for training progress
    position.append((j + 1) / len(eps))
  print(f'  chain {i:2d}: {len(data["is_first"]):6d} steps  {len(eps):5d} episodes')

lengths = np.array(lengths)
complete = np.array(complete)
success = np.array(success)
position = np.array(position)

print(f'\ntotal steps: {total_steps}')
print(f'episodes:    {len(lengths)}  '
      f'complete={int(complete.sum())}  truncated={int((~complete).sum())}')
print(f'successful:  {int(success.sum())}  '
      f'({100 * success.mean():.1f}% reached the goal)')


def percentiles(label, values):
  if not len(values):
    print(f'{label:<22} (none)')
    return
  cells = '  '.join(f'p{p}={int(np.percentile(values, p)):4d}' for p in PCTS)
  print(f'{label:<22} n={len(values):6d}  {cells}')


print('\nepisode length distribution')
percentiles('all', lengths)
percentiles('successful', lengths[success])
percentiles('failed/truncated', lengths[~success])
percentiles('last 10% of chain', lengths[position > 0.9])

# The decision table. An episode of length L yields L // W non-overlapping
# windows, or L - W + 1 with stride 1. Overlapping gives far more samples but
# they are highly correlated, so both are worth seeing.
print('\nwindow viability')
print(f'{"W":>4} {"eps>=W":>8} {"%eps":>6} {"non-ovlp":>9} {"stride1":>9}'
      f' | {"succ>=W":>8} {"%succ":>6}')
for w in [4, 8, 16, 32, 64]:
  fits = lengths >= w
  sfits = lengths[success] >= w
  nonov = int((lengths[fits] // w).sum())
  stride = int((lengths[fits] - w + 1).sum())
  pct = 100 * fits.mean()
  spct = 100 * sfits.mean() if len(sfits) else 0.0
  print(f'{w:>4} {int(fits.sum()):>8} {pct:>5.1f}% {nonov:>9} {stride:>9}'
        f' | {int(sfits.sum()):>8} {spct:>5.1f}%')

print('\nThe right-hand columns are the ones that matter: a window length that '
      'keeps most *successful* episodes is one whose dataset actually contains '
      'the behaviour we want goals to describe.')
