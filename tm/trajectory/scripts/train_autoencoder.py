"""Train the trajectory autoencoder. NOT YET IMPLEMENTED.

PIPELINE POSITION
  <logdir>/states/ -> dataset.py -> autoencoder.py -> [this script]
  Needs a GPU but not the Dreamer checkpoint: it reads encoded states, so the
  world model plays no part. Will need cluster/train_autoencoder.sbatch.

PLANNED ORDER, deliberately incremental
  1. Overfit one batch of ~32 windows -- proves the model and loss can learn
     anything at all before data volume enters the picture.
  2. Train on all chains with two held out for validation, and confirm
     held-out reconstruction improves rather than only training loss.
  3. Check embeddings have non-zero variance -- a collapsed encoder that maps
     everything to one point can still show a falling loss.
  4. Check whether DoorKey phases group in embedding space.

SAMPLING, DECIDED
  Failures time out at 501 steps while successes finish in ~28, so a
  step-uniform sample is roughly 64% failure behaviour. Use
  windows.balanced_weights so each episode carries equal probability, which
  restores the 88% success rate the episodes actually have. Keep the mode a
  config knob (all / balanced / successful-only) rather than a constant: the
  right mixture is a research question, not an obvious default, since failed
  behaviour is still the agent's own past behaviour.

SPLIT, DECIDED
  Hold out whole chains, not random episodes. Chains are independent
  environments with different seeds, so a held-out chain shares no layouts or
  trajectories. Success rates run 85.9-89.4% across the 16, so no chain is
  unrepresentative. Note this tests generalisation to new layouts and
  trajectories, not to new behaviour -- one policy drove all 16.
"""
