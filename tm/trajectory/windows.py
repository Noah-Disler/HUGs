"""Turn saved replay chunks into episodes, and episodes into windows.

Dreamer's replay buffer writes fixed-size chunks (1024 steps) per environment,
linked into chains by a `succ` pointer in the filename. Episodes are variable
length and do not align with those boundaries, so an episode routinely spans
two chunks. Recovering it therefore means chaining first, splitting second.

Splitting matters beyond tidiness: to encode a sequence through the RSSM you
must start where it was wiped (`is_first`), because that is the only point
where the recurrent state is known. Begin partway into an episode and every
latent you produce is wrong.

Functions here are deliberately split into pure logic (parsing, chaining,
segmenting) and I/O (globbing, loading), so the logic can be unit tested
without any replay data.

PIPELINE POSITION
  Shared library, no side effects. Used by scripts/encode_states.py,
  scripts/episode_stats.py, and (once written) dataset.py.
  Covered by tests/test_windows.py -- 23 tests, runnable locally with
  `python -m pytest tm/trajectory/tests/test_windows.py`.

KEY GUARANTEE
  window_index only emits windows generated inside episode bounds, so no
  window can straddle a reset. That is structural, not a check applied
  afterwards, and test_no_window_ever_crosses_a_reset pins it.
"""

import collections
import pathlib

import numpy as np

# One saved chunk. `succ` names the chunk continuing after this one; the last
# chunk of a chain points at a successor that was created but never written
# (empty chunks are not saved), so that dangling id terminates the walk.
ChunkRef = collections.namedtuple('ChunkRef', 'name time uuid succ length')

# Half-open [start, stop) into a chain. `complete` is False for an episode
# that was still running when the buffer was flushed -- valid to encode, since
# it does start at a reset, but truncated.
Episode = collections.namedtuple('Episode', 'start stop complete')

# Parallel int arrays, one entry per window. `start`/`stop` are absolute
# indices into the chain; `episode` says which episode the window came from.
WindowIndex = collections.namedtuple('WindowIndex', 'episode start stop')


def parse_chunk_name(name):
  """Split `{time}-{uuid}-{succ}-{length}.npz` into its fields."""
  stem = name[:-len('.npz')] if name.endswith('.npz') else name
  time, uuid, succ, length = stem.split('-')
  return ChunkRef(name, time, uuid, succ, int(length))


def build_chains(names):
  """Order chunk filenames into per-environment chains.

  A chunk is the head of a chain when its uuid is nobody's successor. Each
  chain is then walked by following `succ`. Returns a list of lists of
  filenames, ordered oldest chain first.

  Assumes no chunk was evicted mid-chain, which holds whenever the run stayed
  under the replay capacity. A gap would make the chunk after it look like a
  head, silently splitting one chain in two.
  """
  refs = [parse_chunk_name(n) for n in names]
  by_uuid = {r.uuid: r for r in refs}
  assert len(by_uuid) == len(refs), 'duplicate chunk uuids'
  successors = {r.succ for r in refs}
  heads = [r for r in refs if r.uuid not in successors]

  chains = []
  for head in sorted(heads, key=lambda r: r.time):
    chain, seen, ref = [], set(), head
    while ref is not None:
      assert ref.uuid not in seen, f'cycle in chunk chain at {ref.uuid}'
      seen.add(ref.uuid)
      chain.append(ref.name)
      ref = by_uuid.get(ref.succ)
    chains.append(chain)
  return chains


def episode_bounds(is_first, is_last=None):
  """Segment a chain's `is_first` stream into episodes.

  Any steps before the first reset are dropped: their episode began in data we
  do not have, so the RSSM state at that point is unknown and they cannot be
  encoded correctly. The final episode runs to the end of the chain and is
  marked incomplete unless `is_last` says it finished.
  """
  starts = np.flatnonzero(is_first)
  total = len(is_first)
  episodes = []
  for i, start in enumerate(starts):
    stop = int(starts[i + 1]) if i + 1 < len(starts) else total
    complete = bool(is_last[stop - 1]) if is_last is not None else False
    episodes.append(Episode(int(start), stop, complete))
  return episodes


def slice_windows(length, window, stride=None):
  """Offsets of every window of `window` steps that fits inside `length`.

  Stride defaults to the window size, giving non-overlapping windows. A
  smaller stride yields far more samples but they overlap heavily and are
  correlated, so that is a deliberate choice rather than the default.
  """
  stride = window if stride is None else stride
  assert window > 0 and stride > 0, (window, stride)
  if length < window:
    return np.empty(0, int)
  return np.arange(0, length - window + 1, stride)


def window_index(episodes, window, stride=None):
  """Every window that fits inside any episode, as arrays.

  `start` and `stop` are absolute indices into the chain, so a caller slices
  states directly without reference to the episode. `episode` records which
  episode each window came from, which is what makes balanced sampling
  possible.

  Because windows are generated inside episode bounds, none can straddle a
  reset -- the guarantee is structural rather than checked afterwards.
  """
  episode, start, stop = [], [], []
  for i, ep in enumerate(episodes):
    for offset in slice_windows(ep.stop - ep.start, window, stride):
      episode.append(i)
      start.append(ep.start + offset)
      stop.append(ep.start + offset + window)
  return WindowIndex(
      np.array(episode, int), np.array(start, int), np.array(stop, int))


def balanced_weights(index):
  """Sampling weights that make every episode equally likely.

  Windows are cut from steps, not episodes, so a long episode contributes
  proportionally more of them. In this replay failures time out at 501 steps
  while successes finish in ~28, so uniform sampling over windows draws
  roughly two thirds of its batches from failures despite failures being only
  12% of episodes. Weighting each window by the inverse of its episode's
  window count removes that, leaving the episode mix the data actually has.
  """
  counts = collections.Counter(index.episode.tolist())
  weights = np.array([1.0 / counts[e] for e in index.episode.tolist()])
  return weights / weights.sum()


def episode_returns(reward, episodes):
  """Total reward per episode, for filtering on success."""
  return np.array([reward[e.start:e.stop].sum() for e in episodes])


def list_chunks(directory):
  """Filenames of every saved chunk in a replay directory."""
  return sorted(p.name for p in pathlib.Path(directory).glob('*.npz'))


def load_chain(directory, chain, keys):
  """Concatenate `keys` across one chain, oldest chunk first.

  `np.load` on an npz is lazy, so keys that are not requested are never
  decompressed. Skipping `image` makes a chain roughly ten times cheaper to
  read, which matters when only the episode structure is needed.
  """
  directory = pathlib.Path(directory)
  parts = {k: [] for k in keys}
  for name in chain:
    data = np.load(directory / name)
    for key in keys:
      parts[key].append(data[key])
  return {k: np.concatenate(v, 0) for k, v in parts.items()}
