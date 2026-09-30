"""Unit tests for chunk chaining and episode segmentation.

Pure logic only -- fabricated filenames and hand-built boolean arrays, no
replay data, no checkpoint, no GPU. These run anywhere numpy does.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import windows  # noqa: E402

# A real filename from the DoorKey-8x8 run, so the parser is pinned against
# actual data rather than against my idea of the format.
REAL = ('20260903T154747F927611-7whQ8RPgkxeeHwwZKLoiAs'
        '-14S6rsHxTQbw2zytHcDQol-1024.npz')

# Two chains whose alphabetical order interleaves them. Chain A is
# aaa -> bbb -> ccc, chain B is ddd -> eee; both end on a successor that was
# never written. Sorting these names gives aaa, ddd, bbb, eee, ccc -- so any
# implementation that sorts instead of following `succ` fails below.
CHAIN_NAMES = [
    '0001-aaa-bbb-1024.npz',
    '0003-bbb-ccc-1024.npz',
    '0005-ccc-nosuchchunk-512.npz',
    '0002-ddd-eee-1024.npz',
    '0004-eee-alsomissing-100.npz',
]


def test_parse_splits_fields_in_the_right_order():
  # Catches a swap of uuid and succ, which would silently reverse every chain.
  ref = windows.parse_chunk_name(REAL)
  assert ref.time == '20260903T154747F927611'
  assert ref.uuid == '7whQ8RPgkxeeHwwZKLoiAs'
  assert ref.succ == '14S6rsHxTQbw2zytHcDQol'
  assert ref.length == 1024
  assert ref.name == REAL


def test_parse_length_is_an_integer():
  # Catches a missing int(): '1024' compares and concatenates without error,
  # so this stays invisible until some arithmetic quietly misbehaves.
  ref = windows.parse_chunk_name(REAL)
  assert isinstance(ref.length, int)


def test_parse_tolerates_a_missing_extension():
  # The parsed fields must not depend on the extension. `name` is excluded
  # deliberately: it echoes back whatever was passed in, because load_chain
  # uses it to open the file, so a stem in means a stem out.
  stem = windows.parse_chunk_name(REAL[:-4])
  full = windows.parse_chunk_name(REAL)
  fields = lambda r: (r.time, r.uuid, r.succ, r.length)
  assert fields(stem) == fields(full)


def test_chains_follow_succ_not_alphabetical_order():
  # The central test. Chunks are written by 16 environments interleaved, so
  # filename order is not recording order. Sorting looks like it works on
  # small samples and corrupts every episode that spans a chunk boundary.
  chains = windows.build_chains(CHAIN_NAMES)
  assert len(chains) == 2
  assert chains[0] == [
      '0001-aaa-bbb-1024.npz',
      '0003-bbb-ccc-1024.npz',
      '0005-ccc-nosuchchunk-512.npz',
  ]
  assert chains[1] == [
      '0002-ddd-eee-1024.npz',
      '0004-eee-alsomissing-100.npz',
  ]


def test_chains_terminate_on_a_dangling_successor():
  # Every chain ends pointing at a successor that was created but never
  # written, because empty chunks are not saved. The walk must stop there
  # rather than raise KeyError.
  chains = windows.build_chains(CHAIN_NAMES)
  assert chains[0][-1] == '0005-ccc-nosuchchunk-512.npz'


def test_every_chunk_appears_exactly_once():
  # Catches head detection that drops or duplicates chunks -- either way you
  # lose or double-count data, with no error raised.
  chains = windows.build_chains(CHAIN_NAMES)
  flat = [name for chain in chains for name in chain]
  assert sorted(flat) == sorted(CHAIN_NAMES)


def test_chain_order_is_deterministic():
  # Reproducibility: the same directory must always yield the same chain
  # order, whatever order the filesystem hands back.
  a = windows.build_chains(CHAIN_NAMES)
  b = windows.build_chains(list(reversed(CHAIN_NAMES)))
  assert a == b


def test_episodes_split_on_each_reset():
  # Catches an off-by-one in `stop`, which would make episodes overlap by a
  # step or leave a gap between them.
  is_first = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0], bool)
  eps = windows.episode_bounds(is_first)
  assert [(e.start, e.stop) for e in eps] == [(0, 5), (5, 10)]


def test_steps_before_the_first_reset_are_dropped():
  # A chain may begin partway through an episode. Those steps cannot be
  # encoded -- the RSSM state where they start is unknown -- so they must not
  # be emitted. Including them would produce plausible but wrong latents.
  is_first = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], bool)
  eps = windows.episode_bounds(is_first)
  assert [(e.start, e.stop) for e in eps] == [(3, 10)]


def test_no_resets_gives_no_episodes():
  assert windows.episode_bounds(np.zeros(10, bool)) == []


def test_completeness_comes_from_the_last_step_of_each_episode():
  # Catches indexing is_last[stop] instead of is_last[stop - 1], which reads
  # the *next* episode's first step, or runs off the end of the array.
  is_first = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0], bool)
  is_last = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], bool)
  eps = windows.episode_bounds(is_first, is_last)
  assert eps[0].complete is True    # ended properly at index 4
  assert eps[1].complete is False   # still running when the buffer flushed


def test_episodes_tile_the_stream_without_gaps_or_overlaps():
  # The invariant the whole module exists to provide: each episode contains
  # exactly one reset, at its own first step. That is what guarantees no
  # window sliced inside an episode can ever straddle a reset.
  rng = np.random.default_rng(0)
  is_first = rng.random(200) < 0.1
  is_first[0] = True
  eps = windows.episode_bounds(is_first)
  for prev, nxt in zip(eps, eps[1:]):
    assert prev.stop == nxt.start
  assert eps[-1].stop == len(is_first)
  for ep in eps:
    assert is_first[ep.start]
    assert not is_first[ep.start + 1:ep.stop].any()


def test_windows_do_not_overlap_by_default():
  # Default stride is the window size. Overlapping samples are correlated, so
  # they should be opt-in rather than something you get without asking.
  assert list(windows.slice_windows(10, 4)) == [0, 4]
  assert list(windows.slice_windows(8, 4)) == [0, 4]


def test_stride_controls_overlap():
  assert list(windows.slice_windows(10, 4, stride=2)) == [0, 2, 4, 6]
  assert list(windows.slice_windows(10, 4, stride=1)) == [0, 1, 2, 3, 4, 5, 6]


def test_episodes_shorter_than_the_window_yield_nothing():
  # This is the quiet data-loss case: at W=32 more than half the episodes in
  # the real replay drop out entirely, so it must be an empty result rather
  # than a truncated or padded window.
  assert len(windows.slice_windows(15, 16)) == 0
  assert list(windows.slice_windows(16, 16)) == [0]


def test_window_index_uses_absolute_indices():
  # start/stop index the chain, not the episode, so a caller can slice states
  # directly. An episode-relative offset here would silently read the wrong
  # part of the array for every episode after the first.
  eps = [windows.Episode(100, 120, True)]
  idx = windows.window_index(eps, 8)
  assert list(idx.start) == [100, 108]
  assert list(idx.stop) == [108, 116]
  assert list(idx.episode) == [0, 0]


def test_no_window_ever_crosses_a_reset():
  # The invariant the module exists for. A window may begin on a reset -- that
  # is its episode's first step -- but must contain no reset anywhere else.
  rng = np.random.default_rng(1)
  is_first = rng.random(400) < 0.08
  is_first[0] = True
  eps = windows.episode_bounds(is_first)
  idx = windows.window_index(eps, 8, stride=1)
  assert len(idx.start) > 0
  for start, stop in zip(idx.start, idx.stop):
    assert not is_first[start + 1:stop].any()


def test_windows_stay_inside_their_episode():
  rng = np.random.default_rng(2)
  is_first = rng.random(400) < 0.08
  is_first[0] = True
  eps = windows.episode_bounds(is_first)
  idx = windows.window_index(eps, 8, stride=3)
  for episode, start, stop in zip(idx.episode, idx.start, idx.stop):
    assert eps[episode].start <= start
    assert stop <= eps[episode].stop


def test_balanced_weights_give_each_episode_equal_mass():
  # A 100-step episode produces far more windows than a 20-step one, so
  # uniform sampling over windows is dominated by the long one. Weighting
  # should leave each episode with the same total probability.
  eps = [windows.Episode(0, 100, True), windows.Episode(100, 120, True)]
  idx = windows.window_index(eps, 10)
  weights = windows.balanced_weights(idx)
  assert np.isclose(weights.sum(), 1.0)
  mass = [weights[idx.episode == i].sum() for i in range(len(eps))]
  assert np.allclose(mass, 0.5)


def test_uniform_sampling_really_is_skewed_without_weights():
  # Pins the problem the weights solve, using the real shape of the DoorKey
  # replay: a failure times out at 501 steps, a success finishes in 28. At
  # W=16 the failure yields 31 windows and the success yields 1, so uniform
  # sampling over windows draws 97% of batches from the failure despite the
  # two episodes being equally numerous.
  failure = windows.Episode(0, 501, True)
  success = windows.Episode(501, 529, True)
  idx = windows.window_index([failure, success], 16)
  assert (idx.episode == 0).sum() == 31
  assert (idx.episode == 1).sum() == 1
  assert (idx.episode == 0).mean() > 0.95

  # and the weights restore parity
  weights = windows.balanced_weights(idx)
  assert np.isclose(weights[idx.episode == 0].sum(), 0.5)
  assert np.isclose(weights[idx.episode == 1].sum(), 0.5)


def test_episode_returns_sums_within_bounds():
  reward = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.5])
  eps = [windows.Episode(0, 3, True), windows.Episode(3, 6, True)]
  assert list(windows.episode_returns(reward, eps)) == [1.0, 0.5]


def test_load_chain_concatenates_in_chain_order(tmp_path):
  # Chain order here is the reverse of alphabetical order, so an
  # implementation that sorts produces [3, 4, 1, 2] instead of [1, 2, 3, 4].
  np.savez(tmp_path / '0009-aaa-bbb-2.npz', x=np.array([1, 2]),
           y=np.array([10, 20]))
  np.savez(tmp_path / '0001-bbb-gone-2.npz', x=np.array([3, 4]),
           y=np.array([30, 40]))
  chain = ['0009-aaa-bbb-2.npz', '0001-bbb-gone-2.npz']
  out = windows.load_chain(tmp_path, chain, ['x'])
  assert list(out['x']) == [1, 2, 3, 4]


def test_load_chain_returns_only_requested_keys(tmp_path):
  # Not cosmetic: skipping `image` is what makes reading a chain ~10x cheaper,
  # since npz decompresses lazily and pixels are 83% of every chunk.
  np.savez(tmp_path / '0001-aaa-gone-2.npz', x=np.array([1, 2]),
           y=np.array([10, 20]))
  out = windows.load_chain(tmp_path, ['0001-aaa-gone-2.npz'], ['x'])
  assert set(out) == {'x'}
