"""Serve windows of model states to the trajectory autoencoder.

The bridge between what scripts/encode_states.py wrote and what the
autoencoder wants. It memory-maps the encoded states, applies the split and
the competence filter, builds a window index, weights it, and gathers
[B, W, 640] batches.

    <logdir>/states/chain00..15/   ->  [this]  ->  [B, 16, 640]
      state.npy   memory-mapped
      index.npz   episode table

PIPELINE POSITION
  scripts/encode_states.py -> [this] -> autoencoder.py, driven by
  scripts/train_autoencoder.py. Pure numpy: no JAX, no GPU, no checkpoint,
  so it can be exercised without a cluster job.

DECISIONS BAKED IN HERE, AND WHY
  Normalisation statistics are computed from training chains only. Using all
  16 would leak validation statistics into training -- a small leak, but free
  to avoid.

  Validation is a fixed, deterministic set of windows; only training samples
  with replacement. Otherwise validation loss wobbles between epochs for
  sampling reasons and improvement cannot be distinguished from noise.

  Episode ids are offset to be unique across chains before weighting. Without
  that, episode 0 of chain 0 and episode 0 of chain 7 look like one episode
  and the balancing is quietly wrong.

  Sampling defaults to episode-balanced. Failures time out at 501 steps while
  successes finish in ~28, so drawing windows uniformly gives ~64% failure
  behaviour despite failures being 12% of episodes. `balanced=False` restores
  the raw step-uniform mixture, which is a legitimate thing to compare
  against, not a bug.
"""

import pathlib

import numpy as np

import windows

DEFAULT_VAL_CHAINS = (14, 15)


def default_stride(window):
  """Half the window, i.e. 50% overlap.

  Non-overlapping windows at W=16 give only ~25k training samples against a
  16 x 640 = 10,240-dimensional input, which is thin. Halving the stride
  roughly doubles the count. The samples are correlated -- neighbours share
  half their states -- so this buys less than the nominal count suggests, and
  it is a knob rather than a default to accept blindly: pass stride=window for
  independent samples, or stride=1 for the maximum.
  """
  return max(1, window // 2)


class StateWindows:
  """Windows of encoded model states drawn from a set of chains."""

  def __init__(
      self, root, chain_ids, window=16, stride=None, successful_only=False,
      balanced=True):
    self.root = pathlib.Path(root)
    self.window = window
    self.stride = default_stride(window) if stride is None else stride
    self.balanced = balanced
    self.states = []
    chain, episode, start, stop = [], [], [], []
    offset = 0  # makes episode ids unique across chains

    for slot, cid in enumerate(chain_ids):
      path = self.root / f'chain{cid:02d}'
      # mmap so the 1.5 GB never lands in RAM; only touched windows page in
      self.states.append(np.load(path / 'state.npy', mmap_mode='r'))
      index = np.load(path / 'index.npz')
      episodes = [
          windows.Episode(int(a), int(b), bool(c)) for a, b, c in zip(
              index['episode_start'], index['episode_stop'],
              index['episode_complete'])]
      if successful_only:
        keep = index['episode_return'] > 0
        episodes = [e for e, k in zip(episodes, keep) if k]
      idx = windows.window_index(episodes, window, self.stride)
      chain.append(np.full(len(idx.start), slot, np.int32))
      episode.append(idx.episode + offset)
      start.append(idx.start)
      stop.append(idx.stop)
      offset += len(episodes)

    # Taken from the data rather than a constant, so a mismatch between what
    # encode_states.py wrote and what we expect fails loudly here.
    self.dim = self.states[0].shape[-1]
    assert all(s.shape[-1] == self.dim for s in self.states), 'ragged dims'
    self.chain = np.concatenate(chain)
    self.index = windows.WindowIndex(
        np.concatenate(episode), np.concatenate(start), np.concatenate(stop))
    self.weights = (
        windows.balanced_weights(self.index) if balanced else
        np.full(len(self.index.start), 1.0 / len(self.index.start)))
    self.mean = np.zeros(self.dim, np.float32)
    self.std = np.ones(self.dim, np.float32)

  def __len__(self):
    return len(self.index.start)

  def gather(self, ids):
    """Windows for the given ids, as [len(ids), window, dim]."""
    out = np.empty((len(ids), self.window, self.dim), np.float32)
    for i, j in enumerate(ids):
      out[i] = self.states[self.chain[j]][self.index.start[j]:
                                          self.index.stop[j]]
    return (out - self.mean) / self.std

  def sample(self, rng, size):
    """A weighted random batch, with replacement. For training."""
    return self.gather(rng.choice(len(self), size, p=self.weights))

  def fixed(self, size, seed=0):
    """A deterministic batch. For validation and the overfit check.

    Shuffled rather than taken in order, so a fixed batch is not all drawn
    from one episode or one chain.
    """
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(self))[:size]
    return self.gather(np.sort(ids))

  def fit_normalizer(self, rng, samples=50_000):
    """Per-dimension mean and std, estimated from a subsample of steps.

    Sampled at the step level rather than the window level: 50k windows would
    be ~4 GB to materialise, while 50k steps is ~128 MB and is ample for
    640 statistics.
    """
    per_chain = max(1, samples // len(self.states))
    rows = []
    for state in self.states:
      pick = rng.choice(len(state), min(per_chain, len(state)), replace=False)
      rows.append(np.asarray(state[np.sort(pick)], np.float32))
    flat = np.concatenate(rows, 0)
    self.mean = flat.mean(0)
    self.std = flat.std(0) + 1e-6
    return self.mean, self.std


def load(root, val_chains=DEFAULT_VAL_CHAINS, window=16, stride=None,
         successful_only=False, balanced=True, normalize=True, seed=0):
  """Build the training and validation sets.

  Whole chains are held out rather than random episodes: chains are separate
  environments with different seeds, so a held-out chain shares no layouts and
  no trajectories. Note this tests generalisation to new layouts and
  trajectories, not to new behaviour -- one policy drove all sixteen.
  """
  root = pathlib.Path(root)
  found = sorted(int(p.name[5:]) for p in root.glob('chain*') if p.is_dir())
  assert found, f'no chain directories under {root}'
  val_chains = tuple(c for c in val_chains if c in found)
  train_chains = tuple(c for c in found if c not in val_chains)
  assert train_chains and val_chains, (train_chains, val_chains)

  kw = dict(window=window, stride=stride, successful_only=successful_only)
  train = StateWindows(root, train_chains, balanced=balanced, **kw)
  # Validation is never balanced-sampled: it is enumerated deterministically,
  # so weighting it would have no effect and only invite confusion.
  val = StateWindows(root, val_chains, balanced=False, **kw)

  if normalize:
    # Training chains only, so validation statistics do not leak into training.
    mean, std = train.fit_normalizer(np.random.default_rng(seed))
    val.mean, val.std = mean, std

  return train, val
