"""Unit tests for the window loader.

Builds fake chain directories in a tmp dir -- small state.npy arrays with a
matching episode table -- so nothing here needs the cluster, the checkpoint,
or the real 1.5 GB of encoded states.

The state dimension is 8 rather than the real 640 purely to keep the fixtures
cheap; dataset.py reads the width from the data, so the tests exercise the
same code path.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dataset  # noqa: E402
import windows  # noqa: E402

DIM = 8


def write_chain(root, cid, episodes, offset=0.0, seed=0):
  """Create one fake chain. `episodes` is a list of (length, return).

  States are standard normal plus `offset`, which lets a test give the
  validation chains a distinguishable scale.
  """
  rng = np.random.default_rng(seed)
  starts, stops = [], []
  position = 0
  for length, _ in episodes:
    starts.append(position)
    stops.append(position + length)
    position += length
  state = rng.normal(size=(position, DIM)).astype(np.float32) + offset
  path = pathlib.Path(root) / f'chain{cid:02d}'
  path.mkdir(parents=True)
  np.save(path / 'state.npy', state)
  np.savez(
      path / 'index.npz',
      episode_start=np.array(starts, np.int32),
      episode_stop=np.array(stops, np.int32),
      episode_complete=np.ones(len(episodes), bool),
      episode_return=np.array([r for _, r in episodes], np.float32))
  return state


def build(root, n_chains=4, episodes=((16, 1.0), (8, 1.0)), **kw):
  for cid in range(n_chains):
    write_chain(root, cid, episodes, seed=cid, **kw)
  return root


# --- splitting --------------------------------------------------------------

def test_split_holds_out_exactly_the_named_chains(tmp_path):
  build(tmp_path)
  train, val = dataset.load(
      tmp_path, val_chains=(2, 3), window=4, stride=4, normalize=False)
  assert len(train.states) == 2
  assert len(val.states) == 2


def test_val_chains_outside_the_data_are_ignored(tmp_path):
  # load() defaults to chains 14 and 15, which will not exist in a short run.
  # It should hold out whatever of them is present rather than crashing.
  build(tmp_path, n_chains=4)
  train, val = dataset.load(
      tmp_path, val_chains=(3, 99), window=4, stride=4, normalize=False)
  assert len(train.states) == 3
  assert len(val.states) == 1


# --- window counts ----------------------------------------------------------

def test_window_counts_match_the_episode_lengths(tmp_path):
  # Per chain: a 16-step episode gives 4 non-overlapping windows of 4
  # (offsets 0, 4, 8, 12) and an 8-step episode gives 2 (offsets 0, 4).
  build(tmp_path, n_chains=4)
  train, val = dataset.load(
      tmp_path, val_chains=(3,), window=4, stride=4, normalize=False)
  assert len(train) == 3 * 6
  assert len(val) == 1 * 6


def test_default_stride_is_half_the_window(tmp_path):
  # 50% overlap: the 16-step episode now yields offsets 0,2,4,6,8,10,12 = 7
  # and the 8-step one 0,2,4 = 3, so 10 per chain rather than 6.
  assert dataset.default_stride(16) == 8
  assert dataset.default_stride(4) == 2
  build(tmp_path, n_chains=2)
  train, _ = dataset.load(
      tmp_path, val_chains=(1,), window=4, normalize=False)
  assert len(train) == 10


# --- the episode-id offset, which is easy to get silently wrong -------------

def test_episode_ids_are_unique_across_chains(tmp_path):
  # Without the per-chain offset, episode 0 of chain 0 and episode 0 of
  # chain 1 collide, and the balancing below silently treats them as one.
  build(tmp_path, n_chains=3)
  train, _ = dataset.load(
      tmp_path, val_chains=(2,), window=4, stride=4, normalize=False)
  assert len(np.unique(train.index.episode)) == 2 * 2  # 2 chains x 2 episodes


def test_balanced_weights_equalise_episodes_across_chains(tmp_path):
  # Episodes of 16 and 8 steps give 4 and 2 windows, so unweighted sampling
  # favours the long one 2:1. Each of the four episodes should carry 0.25.
  build(tmp_path, n_chains=3)
  train, _ = dataset.load(
      tmp_path, val_chains=(2,), window=4, stride=4, normalize=False)
  assert np.isclose(train.weights.sum(), 1.0)
  for episode in np.unique(train.index.episode):
    mass = train.weights[train.index.episode == episode].sum()
    assert np.isclose(mass, 0.25), (episode, mass)


def test_unbalanced_sampling_is_available_and_differs(tmp_path):
  # The raw step-uniform mixture is a legitimate comparison point, not a bug.
  build(tmp_path, n_chains=3)
  train, _ = dataset.load(
      tmp_path, val_chains=(2,), window=4, stride=4, balanced=False,
      normalize=False)
  assert np.allclose(train.weights, 1.0 / len(train))


# --- filtering --------------------------------------------------------------

def test_successful_only_drops_zero_return_episodes(tmp_path):
  build(tmp_path, n_chains=2, episodes=((16, 1.0), (8, 0.0)))
  train, _ = dataset.load(
      tmp_path, val_chains=(1,), window=4, stride=4, normalize=False)
  kept, _ = dataset.load(
      tmp_path, val_chains=(1,), window=4, stride=4, successful_only=True,
      normalize=False)
  assert len(train) == 6     # 4 from the 16-step episode, 2 from the 8-step
  assert len(kept) == 4      # only the successful one survives


# --- gathering --------------------------------------------------------------

def test_gather_returns_the_underlying_states(tmp_path):
  state = write_chain(tmp_path, 0, ((16, 1.0),))
  write_chain(tmp_path, 1, ((16, 1.0),), seed=1)
  train, _ = dataset.load(
      tmp_path, val_chains=(1,), window=4, stride=4, normalize=False)
  batch = train.gather(np.arange(len(train)))
  assert batch.shape == (4, 4, DIM)
  for i in range(len(train)):
    begin = train.index.start[i]
    assert np.array_equal(batch[i], state[begin:begin + 4])


def test_no_window_spans_two_episodes(tmp_path):
  # The guarantee windows.py provides, re-checked after the chain offsetting
  # and concatenation that dataset.py does on top of it.
  build(tmp_path, n_chains=2, episodes=((16, 1.0), (8, 1.0), (12, 1.0)))
  train, _ = dataset.load(
      tmp_path, val_chains=(1,), window=4, stride=2, normalize=False)
  bounds = [(0, 16), (16, 24), (24, 36)]
  for begin, end in zip(train.index.start, train.index.stop):
    assert any(lo <= begin and end <= hi for lo, hi in bounds), (begin, end)


# --- determinism ------------------------------------------------------------

def test_fixed_batches_are_reproducible(tmp_path):
  # A 256-step episode gives 64 windows, so two seeds picking the same 4 is
  # vanishingly unlikely. With only a handful of windows this test would be
  # flaky rather than wrong.
  build(tmp_path, n_chains=2, episodes=((256, 1.0),))
  train, _ = dataset.load(
      tmp_path, val_chains=(1,), window=4, stride=4, normalize=False)
  assert np.array_equal(train.fixed(4), train.fixed(4))
  assert not np.array_equal(train.fixed(4, seed=0), train.fixed(4, seed=1))


def test_sampling_respects_the_requested_size(tmp_path):
  build(tmp_path, n_chains=2)
  train, _ = dataset.load(
      tmp_path, val_chains=(1,), window=4, stride=4, normalize=False)
  batch = train.sample(np.random.default_rng(0), 32)
  assert batch.shape == (32, 4, DIM)  # with replacement, so size > len is fine


# --- normalisation, and the leak it must not create -------------------------

def test_normalisation_statistics_come_from_training_chains_only(tmp_path):
  # The validation chain is shifted by +100. If its statistics leaked into
  # the normaliser the training mean would be dragged far from zero.
  write_chain(tmp_path, 0, ((64, 1.0),), offset=0.0, seed=0)
  write_chain(tmp_path, 1, ((64, 1.0),), offset=0.0, seed=1)
  write_chain(tmp_path, 2, ((64, 1.0),), offset=100.0, seed=2)
  train, val = dataset.load(
      tmp_path, val_chains=(2,), window=4, stride=4, normalize=True)
  assert np.abs(train.mean).max() < 5.0
  assert np.array_equal(train.mean, val.mean)
  assert np.array_equal(train.std, val.std)


def test_normalised_training_batches_are_roughly_standardised(tmp_path):
  build(tmp_path, n_chains=3, episodes=((256, 1.0),))
  train, _ = dataset.load(
      tmp_path, val_chains=(2,), window=4, stride=4, normalize=True)
  batch = train.gather(np.arange(len(train)))
  flat = batch.reshape(-1, DIM)
  assert np.abs(flat.mean(0)).max() < 0.3
  assert np.abs(flat.std(0) - 1.0).max() < 0.3


def test_normalisation_can_be_disabled(tmp_path):
  build(tmp_path, n_chains=2)
  train, _ = dataset.load(
      tmp_path, val_chains=(1,), window=4, stride=4, normalize=False)
  assert np.array_equal(train.mean, np.zeros(DIM, np.float32))
  assert np.array_equal(train.std, np.ones(DIM, np.float32))
