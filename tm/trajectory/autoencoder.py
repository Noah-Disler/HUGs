"""Trajectory encoder, decoder and reconstruction loss. NOT YET IMPLEMENTED.

The core of the HUGs proposal. Compresses a window of RSSM model states into a
single trajectory embedding, then reconstructs the window from it:

    tau = f_phi(s_[0,n])        encoder:  [W, 640] -> [D]
    s_hat = f_psi(tau)          decoder:  [D] -> [W, 640]

tau is the learned goal representation. If it can reconstruct the window, it
has captured the behavioural structure of that stretch of behaviour, which is
what the proposal means by a goal inferred from the agent's own past.

PIPELINE POSITION
  dataset.py yields [B, W, 640] batches -> [this] -> scripts/train_autoencoder.py

SETTLED BEFORE WRITING THIS
  Input is the full model state s_t = concat(deter, flat(stoch)) = 640-d, not
  z_t alone. DoorKey is partially observable, so whether the key is held lives
  in deter; windows of z alone would omit the phase structure this is meant to
  discover. See the proposal-notation note: the write-up says z_[0,n] but
  means s_[0,n].
  W = 16, chosen from scripts/episode_stats.py -- it retains 96% of successful
  episodes where W=32 retains 40%.

STILL OPEN
  Embedding dimension D. Encoder architecture (MLP over the flattened window,
  1D conv, or a small transformer). Whether to normalise states first -- they
  sit around 0.097 in magnitude and decorrelate within roughly one step, so
  the window is not smooth and there is little trivial structure to exploit.

FIRST MILESTONE
  Overfit a single batch of ~32 windows. If it cannot memorise 32 examples the
  architecture or the loss is wrong, and no amount of data will help.
"""
