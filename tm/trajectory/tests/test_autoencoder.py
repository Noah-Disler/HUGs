"""Unit tests for the trajectory autoencoder. NOT YET IMPLEMENTED.

Written alongside autoencoder.py, following the pattern of test_windows.py:
pure logic on small fabricated tensors, no checkpoint, no replay data, so the
suite stays runnable locally with
`python -m pytest tm/trajectory/tests/ -v`.

WORTH PINNING WHEN THE TIME COMES
  Shapes survive a round trip: [B, W, 640] -> [B, D] -> [B, W, 640].
  The embedding is genuinely a bottleneck -- encoding two different windows
  gives two different embeddings, so a collapsed encoder fails loudly rather
  than quietly producing a falling loss.
  Reconstruction loss is zero when the decoder is handed the true window.
  Nothing leaks across the batch dimension: perturbing one window changes
  only that row's embedding.

THE FIRST REAL CHECK IS NOT A UNIT TEST
  Overfitting a single batch of ~32 windows needs a GPU and real data, so it
  belongs in scripts/train_autoencoder.py. These tests only establish that
  the pieces are wired correctly enough to be worth running.
"""
