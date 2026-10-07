**Where this document conflicts with the research proposal, this document is the source of truth.**

# HUGs — project handoff

Last updated 2026-10-07. Written for a chat that has none of the previous history.

The proposal is at `/Users/noahdisler/Desktop/Masters/Proposal submission stuff/Noah Disler 2465179 Proposal.pdf`. Section 7 says which parts of it still stand.

---

## 0. How to read this

Decisions are sorted into four kinds, and the distinction matters:

| Label | Meaning |
|---|---|
| **SETTLED** | Decided by Noah (and where noted, his supervisor). Don't reopen without new information. |
| **CARRIED OVER** | Decided during the autoencoder phase. The reasoning probably still applies, but nobody has re-confirmed it for the new direction. Confirm before relying on it. |
| **OPEN** | Undecided. Do not treat as decided, however natural a default looks. |
| **OBSOLETE** | Was decided, no longer applies. |

Proposal elements in section 7 are marked **KEPT / CHANGED / DROPPED / UNDECIDED**.

---

## 1. The project

Noah Disler's MSc (Wits, supervisors Prof. Steven James and Prof. Benjamin Rosman). The repo is a fork of danijar's DreamerV3 (`dreamerv3/`, `embodied/`). New work lives in `tm/trajectory/`.

The research question, unchanged from the proposal: can a hierarchical model of the agent's own behaviour — long-horizon predictions conditioning shorter-horizon ones, ending in an action prediction — improve long-horizon decision-making when used as an auxiliary signal for Dreamer's actor, compared with reward-driven Dreamer alone? Measured by sample efficiency, success rate, return, and behaviour under sparse or delayed reward; later, transfer to held-out tasks.

---

## 2. The current direction (October 2026)

### What changed and why

In Noah's words, after talking to his supervisor:

> The trajectory autoencoder is unnecessary. The model states s_t already summarise the past (Dreamer designs them to be Markovian), so compressing a window of past states into an embedding adds little. Instead the TM predicts future model states directly and hierarchically: the top level predicts ŝ_{t+K} from s_t, the next level predicts ŝ_{t+K-1} conditioned on that prediction, and so on down to ŝ_{t+1}. The action prediction comes from the finest level. K=16 is a starting point; the horizon is something we'll test, and the spacing between levels isn't settled either.

The reasoning holds up against the code: Dreamer's actor is π(a | s_t) and its critic is v(s_t) — both functions of s_t alone (DreamerV3 paper, p.5: they "benefit from the Markovian representations learned by the recurrent world model"). A window of past states therefore carries no information about the agent's future that s_t lacks.

### SETTLED about the trajectory model (TM)

- No trajectory autoencoder.
- The TM predicts **future model states**, top-down: the coarsest level predicts ŝ_{t+K} from s_t; each finer level predicts a nearer state conditioned on the next-coarser prediction; the finest level predicts ŝ_{t+1}.
- An action prediction comes from the finest level.
- It operates on s_t = concat(deter, flatten(stoch)), 640-d at `size1m` (section 6.1, S3).

### Explicitly OPEN about the TM

- **K.** 16 is a starting point, and the horizon will be tested.
- **Level spacing**, and therefore the number of levels.

### Ambiguous in the description — do not assume either reading

- **Does each finer level also see s_t**, or only the coarser prediction? The proposal's version says yes (f_x(z_0, τ̂_{x+1})). The new description mentions only the prediction.
- **"ŝ_{t+K-1}" read literally means spacing 1**, i.e. K levels (16 at K=16). Spacing is open, so treat that as illustrative. 16/8/4/2/1 is another reading.
- **Deterministic or distributional?** The proposal writes the TM as a distribution (τ̂ ~ p), and its confidence gating depends on that.
- **What the action head sees**: only ŝ_{t+1}, or (s_t, ŝ_{t+1})?
- **What happens near the end of an episode**, where t+K falls past the last step.

---

## 3. What exists and is verified

### 3.1 The baseline DreamerV3 run

| | |
|---|---|
| Command | `sbatch cluster/train_bigbatch.sbatch --configs minigrid size1m --task minigrid_DoorKey-8x8-v0 --script train_eval --run.steps 6e5 --run.save_every 600` |
| Job | 49393 on mscluster72, 49 min, completed cleanly |
| Logdir | `/datasets/ndisler/HUGs/logdir/20260903T154700/` (has `config.yaml`, `ckpt/`, `replay/`, `states/`, `metrics.jsonl`, `scores.jsonl`) |
| Model | `size1m`: deter 512, hidden 64, stoch 32 × 4 classes → s_t is 640-d. 686,585 parameters. |
| Relevant config | `imag_length: 15` (actor's imagination horizon), `horizon: 333` (discount horizon), `batch_length: 20` (minigrid preset), 16 training envs, `max_episode_steps: 500` |
| Environment | Partially observable: the 7×7 egocentric view rendered as RGB (`RGBImgPartialObsWrapper`) and resized to 64×64×3. 7 discrete actions. Reward 1 − 0.9·(steps/500) on success, 0 on failure. |
| Learning curve | step 497,040: score 0.96, length 32 · step 559,088: 0.97, length 21 · step 599,964: 0.97, length 23 |
| **Frozen checkpoint** | `ckpt/20260903T162740F453771/` (pointer file `ckpt/latest`). Saved at ~497k steps, score ~0.96. It is the last periodic save — `train_eval` writes no final checkpoint. |

Scores come from the *exploring* policy. `policy(..., mode='eval')` ignores `mode`, so this codebase has no greedy evaluation, and `eval_replay` holds the same behaviour as `replay`.

### 3.2 The replay

`<logdir>/replay/` — 512 compressed `.npz` chunks named `{time}-{uuid}-{succ}-{length}.npz`. `succ` names the next chunk in the same environment's stream.

| | |
|---|---|
| Structure | 16 chains (one per env) × 32 chunks, 30,968 steps per chain, **495,488 steps** total |
| Keys per chunk | `image` [n,64,64,3] uint8 · `action` [n] int32 · `reward` · `is_first` · `is_last` · `is_terminal` · `stepid` · `dyn/deter` [n,512] · `dyn/stoch` [n,32,4] |
| Episodes | **4,832** (4,816 complete; 16 truncated — one per chain, in flight at the flush). **4,237 successful (87.7%)**. Per-chain success 85.9–89.4%, so the chains are interchangeable. |
| Successful lengths | p10 18 · p25 22 · p50 28 · p75 42 · p90 92 · max 495 |
| Failures | Timeouts at 501 from p10 upward |
| Last 10% of each chain | p10 17 · p50 24 · p90 33 · max 90 — the competent policy |
| **Skew** | Failures are 12% of episodes but **~60% of steps**. Any step-uniform sample is dominated by failure behaviour. |

Lengths count transitions including the reset step, so a timeout is 501.

**Window retention** — the fraction of successful episodes long enough to hold at least one window of length W, and the success rate of the episodes that survive:

| W | successful episodes retained | success rate of survivors |
|---|---|---|
| 8 | 100% | 87.8% |
| 16 | 96.1% | 87.4% |
| 32 | 40.3% | 74.6% |
| 64 | 15.2% | 52.7% |

A horizon-K TM needs windows of length **K+1** (input s_t plus targets up to s_{t+K}), so rerun `episode_stats.py` at W=K+1 for each candidate K. The table was computed for the autoencoder.

### 3.3 The `latents()` hook — verified

Dreamer computes s_t on every training step and discards it; none of `train`/`report`/`policy` returns it. Two additive diffs expose it:

- **`dreamerv3/agent.py`** — `Agent.latents(carry, data)`: `enc` → `dyn.observe` → `feat2tensor`, no losses. Returns `state` [B,T,640], `deter` [B,T,512], `stoch` [B,T,32,4], `logit` [B,T,32,4], `is_first` [B,T]. The carry is threaded between calls. The action shift `prevact[t] = action[t-1]` is copied from `_apply_replay_context`. A `TODO(hugs)` comment there covers online use.
- **`embodied/jax/agent.py`** — `self._latents = transform.apply(nj.pure(self.model.latents), self.train_mesh, (tp, tm, ts, ts), (ts, ts), ...)` next to `_report`, plus a public `latents(carry, data)` next to `report()`. Output sharding is `(ts, ts)`, not `_report`'s `(ts, tm)`. The strict key assertion `report()` uses is deliberately omitted, since hand-built batches lack `consec`/`stepid`.

**Usage** (see `check_latents.py` or `encode_states.py`): load `config.yaml` from the logdir → `dreamer_main.make_agent(config)` → `elements.Checkpoint().load(<version dir>, keys=['agent'])` → `carry = agent.init_report(B)` → `batch = internal.device_put(batch, agent.train_sharded)`, `batch['seed'] = agent._seeds(counter, agent.train_mirrored)` → `carry, outs = agent.latents(carry, batch)`.

**Verification** — `tm/trajectory/scripts/check_latents.py`, submitted with `cluster/check_latents.sbatch`. It re-encodes a slice of replay and compares against the `dyn/deter` that Dreamer recorded during training. Results from 2026-09-30, on the longest episode (56 steps) in the newest chunk:

| check | result |
|---|---|
| Aligned | mean abs error 0.0154 |
| Control: actions shifted by one step | 0.0830, i.e. **5.4× worse**, and growing over time |
| t=0 | 0.0020 in both runs |
| `stoch` sample agreement | 0.692 |
| Batch independence (carry threaded across passes) | error at the handoff 0.85× the error elsewhere. A dropped carry would give roughly 6×. |

For scale: on that chunk `deter` has mean magnitude 0.0971; consecutive steps differ by 0.0694; unrelated steps by 0.1130. The aligned error is about a quarter of a real one-step change.

t=0 is the one exactly checkable value: at a reset the RSSM zeroes `deter`, `stoch` and the action, so `deter[0] = _core(0,0,0)`. It matches the 0.0025 spread among stored reset states. The residual error elsewhere comes from `stoch` being sampled.

**Caveat:** the batch-independence run covered only **one** handoff (`SPLIT=32` on a 56-step slice). `SPLIT` is now 16, which gives three, but that hasn't been run. It costs nothing to let it ride along with the next cluster job.

### 3.4 The encoded states — verified

`tm/trajectory/scripts/encode_states.py` (submitted with `cluster/encode_states.sbatch`) ran every replay step through the frozen checkpoint:

```
<logdir>/states/
  chain00 … chain15/
    state.npy   [30968, 640]   float32, memory-mappable
    logit.npy   [30968, 32, 4] float32
    index.npz   is_first, reward, episode_start, episode_stop,
                episode_complete, episode_return
  meta.json     checkpoint, config, seed, batch_steps, totals
```

- 1.5 GB, encoded in 80 s.
- `deter = state[:, :512]`, `stoch = state[:, 512:]` (flattened 32×4).
- `stoch` is sampled. The seed sequence is fixed (one seed per 256-step pass, counter from 0), and logits are stored for probability targets.
- **Cross-check:** totals (495,488 / 4,832 / 4,237) match `episode_stats.py`, which rebuilds the chains independently.
- **Not stored: actions.** Also not stored: per-step `is_last`/`is_terminal`, though the episode table is enough to tell where each episode ends and whether it succeeded.

**Why re-encode instead of using the `dyn/deter` in the replay?** Those values were written by `policy()` at action-selection time, so each one came from the encoder as it was at that point in training — mixed vintage. Re-encoding through one frozen checkpoint gives a consistent representation.

### 3.5 `windows.py` and `dataset.py`

- **`tm/trajectory/windows.py`** — chains chunks by `succ`; a head is a uuid nobody points at. Segments chains into episodes on `is_first`, dropping leading partials (their RSSM state is unknowable) and flagging the truncated trailing episode. Slices windows **only inside episodes, so no window crosses a reset by construction**. Provides absolute-index window tables, episode-balanced sampling weights, and per-episode returns. 23 tests in `tests/test_windows.py`.
- **`tm/trajectory/dataset.py`** *(uncommitted)* — `StateWindows` memory-maps the state files and splits by chain. It offers a successful-only filter and balanced or uniform weights, offsets episode ids so they're unique across chains, and computes normalisation from training chains only. `sample()` draws weighted batches with replacement; `fixed()` returns deterministic batches. `load(root, val_chains=(14, 15), window=16, stride=None→window//2, ...)` builds the train/val pair. It was written to serve autoencoder windows, and its docstring still says so. 15 tests in `tests/test_dataset.py`.
- **Run the tests locally** with `/opt/anaconda3/bin/python3 -m pytest tm/trajectory/tests/ -v` (38 pass). On the Mac, plain `python3` resolves to Xcode's Python, which has no pytest.

### 3.6 Facts learned the hard way

1. **`save_every` is wall-clock seconds**, not steps (`elements.when.Clock`). The minigrid preset's `1e4` means 2.8 hours.
2. **`train_eval` writes no final checkpoint** when the loop ends. You keep only the last periodic save.
3. **`elements.Checkpoint` keeps one version.** `ckpt/` holds a `latest` pointer and a single timestamped directory.
4. **The bare `cp.load(path)` needs the concrete version directory**, because `exists()` checks for the `done` marker inside it. `Checkpoint(ckpt_dir).load()` resolves `latest` itself.
5. **Replay is flushed to `<logdir>/replay/` only when a checkpoint is saved.** The `replay_train.pkl` inside `ckpt/` is a 4-byte pickled `None`.
6. **Action indexing.** `action[t]` in replay is the action taken at step t, which leads from s_t to s_{t+1}. The RSSM consumes `prevact[t] = action[t-1]`. Getting this off by one is silent: shapes are fine and latents look plausible. `check_latents.py`'s control catches it.
7. **The RSSM wipes `deter`, `stoch` and the action wherever `is_first`**, so a whole chain can be encoded as one continuous stream.
8. **`nj.scan` runs over axis 1**, so arrays are [B, T, ...]. Different pass lengths draw different `z` samples even with the same seed, so compare to ground truth, never test for exact equality.
9. **`_take_outs` upcasts bfloat16 to float32.** The model computes in bfloat16, so ~0.4% relative error is the precision floor.
10. **Dead-GPU nodes** fail in ~10 s with exit 6 and "Unable to determine the device handle". mscluster45 and mscluster50 are now excluded, alongside the original list. Known good: 72, 73, 46. The node list is duplicated across three sbatch files, because `#SBATCH` lines can't use variables.

---

## 4. Inventory under the new direction

| Piece | Path | Verdict | Why |
|---|---|---|---|
| `latents()` hook | `dreamerv3/agent.py`, `embodied/jax/agent.py` | **KEEP** | The TM consumes s_t. Verified. |
| Hook test | `tm/trajectory/scripts/check_latents.py`, `cluster/check_latents.sbatch` | **KEEP** | Regression test whenever the hook changes |
| Run, checkpoint, replay | `<logdir>/` | **KEEP** | The data source |
| Encoded states | `<logdir>/states/`, `scripts/encode_states.py`, `cluster/encode_states.sbatch` | **ADAPT** | States are usable as-is, but **actions are missing**. Either re-encode with actions (~80 s plus job overhead) or have the loader read them from the replay chunks in the same chain order. |
| Chaining and windows | `tm/trajectory/windows.py`, `tests/test_windows.py` | **KEEP** | A window of length K+1 is exactly one TM example. "No window crosses a reset" is exactly "s_{t+K} must be in the same episode as s_t". |
| Episode statistics | `scripts/episode_stats.py`, `cluster/episode_stats.sbatch` | **KEEP** | Rerun at W=K+1 to choose K |
| Loader | `tm/trajectory/dataset.py`, `tests/test_dataset.py` | **ADAPT** | Window becomes K+1; split into input and per-level targets; add actions. The stride default (window//2) likely becomes 1, since every valid t is its own prediction problem. Normalising categorical `stoch` targets needs rethinking. The docstring says "autoencoder". |
| Autoencoder notes | `tm/trajectory/autoencoder.py`, `scripts/train_autoencoder.py`, `tests/test_autoencoder.py` | **OBSOLETE** | Docstring-only design notes that **still read as current plans** — delete or repurpose. Their sampling and split reasoning is preserved in section 6.2. |
| Cluster scripts | `cluster/train_bigbatch.sbatch` etc. | **KEEP** | Environment boilerplate duplicated three times — see section 5. |

---

## 5. What's missing

**Needed before the first offline TM experiment**
1. Decisions O1–O7 and O12–O13 in section 6.3.
2. Actions in the TM data (see section 4, encoded states).
3. A TM-shaped loader: per valid t, input s_t, targets ŝ_{t+k} at each level's horizon, and the action target `action[t]`.
4. The TM itself, in JAX/ninjax (C2), with local unit tests in the style of `test_windows.py`.
5. A TM training script and sbatch. It reads encoded states, so it needs a GPU only for speed, not the checkpoint.
6. Offline evaluation (O7). Comparing against Dreamer's own imagination needs another small hook, to imagine from given states.

**Needed for integration (later)**

7. The confidence measure (O8), the auxiliary actor loss and where it applies (O9), and qualitative decoding of predicted states through Dreamer's decoder, which needs a decoder hook similar to `latents()`.

**Housekeeping**

8. Delete or repurpose the three obsolete autoencoder files. Update `dataset.py`'s docstring.
9. Factor the duplicated sbatch environment setup into `cluster/run_python.sh` (the `view_results.sbatch`/`.sh` pattern), and keep one bad-node list.
10. Run `check_latents` once more to exercise `SPLIT=16`.

**From the proposal's schedule**

11. XLand-MiniGrid isn't set up (Table 4.1, Phase 1 item 2).

---

## 6. Decisions

### 6.1 SETTLED

- **S1. No trajectory autoencoder.** Supervisor and Noah, Oct 2026. s_t is Markovian by design.
- **S2. The TM shape:** hierarchical future-state prediction, top-down, with the action prediction from the finest level (section 2).
- **S3. The representation is s_t = concat(deter, flatten(stoch)), 640-d, written s_t, never z_t.** In Dreamer's notation (paper Eq. 1, p.4) z_t is only the stochastic latent; the model state is s_t ≐ {h_t, z_t}. DoorKey is partially observable: whether the key is held or the door is open lives in h_t (`deter`), so z_t alone would drop it.
- **S4. Offline TM data comes from re-encoding** through the frozen checkpoint (`<logdir>/states/`), never from the `dyn/*` cached in the replay, which is mixed vintage.
- **S5. Infrastructure.** Everything executes on the Wits mscluster via Slurm. Noah runs cluster commands and pastes output; the assistant doesn't SSH in. Code is edited locally and synced with `./sync_to_cluster.sh`.

### 6.2 CARRIED OVER from the autoencoder phase — confirm before relying

- **C1. Offline first:** train and evaluate the TM on encoded states from the frozen checkpoint before putting it in Dreamer's training loop. One moving part at a time, a reproducible dataset, and minutes per experiment instead of a 50-minute Dreamer run.
- **C2. JAX/ninjax, reusing `embodied.jax.nets`.** Not PyTorch: the TM is the component that ends up inside Dreamer's loop, and nothing runs locally anyway.
- **C3. Validation by held-out whole chains**, 14 and 15. The chains are independent environments and none is unrepresentative. This tests new layouts and trajectories, not new behaviour, since one policy drove all 16.
- **C4. Episode-balanced sampling by default**, with all / balanced / successful-only as a knob. For the TM this is now a method decision, not a loading detail — see O6.
- **C5. Normalisation statistics from training chains only; deterministic validation batches.**
- **C6. No split or filter baked into stored data;** apply them at load time.

### 6.3 OPEN — do not treat as decided

**These block the first offline experiment**

- **O1. Horizon K and the level schedule.** Things to weigh:
  - **K=16 is about the same as Dreamer's own explicit imagination horizon (`imag_length: 15`).** Dreamer's value function already bootstraps credit far past that. A claim of "longer horizon than Dreamer" therefore needs K well above 15, or needs to be about temporal coherence rather than reach.
  - **Competent DoorKey-8x8 episodes run 17–33 steps.** At horizon K only the first L−K steps of an episode can be inputs: 12 of 28 for a median success at K=16.
  - Rerun `episode_stats.py` at W=K+1. Also measure the step-level fraction of usable inputs, which hasn't been computed.
- **O2. Conditioning.** Does each finer level see s_t as well as the coarser prediction? Only the adjacent coarser level, or all of them?
- **O3. Output form, loss, and teacher forcing.**
  - Deterministic or distributional output.
  - A natural split is regression on `deter` plus categorical cross-entropy on `stoch` against the stored logits or samples.
  - Do finer levels train on *true* or *predicted* coarser states? True is easier to train, but at use time the levels receive predictions — exposure bias. The proposal's loss conditions on predictions.
- **O4. Termination inside the horizon.** Drop such inputs (what `windows.py` does now), truncate the hierarchy to the levels that fit, or predict a terminal state. If dropped, the last K steps of every episode — the goal approach — are never TM inputs, and the TM is out of distribution there when deployed.
- **O5. The action head.** f_a(ŝ_{t+1}) alone is likely under-determined, because the action depends on where you are as well as where you're going. f_a(s_t, ŝ_{t+1}) is inverse dynamics. The target is `action[t]` (fact 6).
- **O6. What the TM learns from.** The auxiliary loss produces gradient only where the TM's action and the actor's disagree. A TM trained on the agent's own replay disagrees mostly where the policy has *since changed*, so by default the loss drags the actor toward older behaviour — and 60% of replay steps are timeouts.
  - Successful-only or return-weighted data turns this into self-imitation (Oh et al. 2018, Self-Imitation Learning — not cited in the proposal).
  - Training on recent data is another option.
  - This decides what the method *is*, so it belongs with the method, not the loader.
- **O7. Offline evaluation and baselines.**
  - Per-level prediction error against horizon.
  - A copy baseline, ŝ_{t+k} = s_t.
  - **Dreamer's own imagination under its actor from s_t** — Dreamer can already produce s_{t+K} for its current policy. If imagination matches the TM, what does the TM add?
  - Action-prediction accuracy against the actor's own π(a | s_t).
- **O12. Stride.** Every valid t is a separate prediction problem, so stride 1 is natural. Neighbouring examples are highly correlated.
- **O13. Target normalisation** when part of the target is a categorical one-hot.

**These matter at integration**

- **O8. Confidence and gating.** The proposal weights by exp(−β·Var). Var needs a distributional TM or an ensemble; the proposal's risk section already names ensemble disagreement. **Confidence is not quality:** repetitive timeout behaviour could be the most predictable thing in the data.
- **O9. Where the auxiliary loss applies.** Figure 3.2 says at each *imagined* step. The TM trains on posterior states (from observations), while imagined states come from the prior, so there is a distribution shift. Also open: stop-gradient into the RSSM (almost certainly needed), the λ_aux and β schedule, and whether the TM keeps training online.

**These matter for the thesis**

- **O10. Testbed.** Is DoorKey-8x8 long-horizon enough for a hierarchy to show an effect? Dreamer already reaches 0.97 in ~500k steps. DoorKey-16x16 is the obvious harder variant; XLand comes later.
- **O11. Framing against related work and motivation** (section 7, items 9–13). Worth reviewing beyond the proposal's citations:
  - self-imitation learning (Oh et al. 2018)
  - methods that predict an agent's own future states under its policy, such as γ-models (Janner et al. 2020) and successor representations

  These are pointers to check, not claims about their contents.

### 6.4 OBSOLETE

- **W=16 window length** (chosen for autoencoder inputs). Superseded by K, with window length K+1. The retention measurements behind it remain valid input to O1.
- **Autoencoder embedding size, architecture and input normalisation** — moot.
- **"Overfit one batch of 32 windows" as the autoencoder's first milestone.** An analogous sanity check for the TM — overfit a small fixed batch of (s_t, futures) — is probably still wise, but hasn't been decided.

---

## 7. Proposal vs current direction

| # | Proposal element (where) | Status | In one line |
|---|---|---|---|
| 1 | Trajectory encoder and decoder (Eq. 3.1; §3.3.1; Table 4.1 item 3) | **DROPPED** | s_t already summarises the past |
| 2 | TM equations τ̂_n ~ p_n(·\|z_0), τ̂_x ~ p_x(·\|z_0, τ̂_{x+1}) (Eq. 3.1) | **CHANGED** | Targets become future model states; top-down conditioning kept |
| 3 | TM loss — KL to the encoder's posterior (Eq. 3.2) | **DROPPED** | No encoder; replacement loss UNDECIDED (O3) |
| 4 | Action model â_1 = f_a(τ̂_0) (Eq. 3.1) | **CHANGED** | Fed by the finest predicted state; exact inputs UNDECIDED (O5) |
| 5 | z_t as "the RSSM latent state" (throughout) | **CHANGED** | Use s_t ≐ {h_t, z_t} |
| 6 | Confidence-gated auxiliary actor loss (Eq. 3.3) | **KEPT** in form; mechanism **UNDECIDED** | Var[τ̂_0] has no direct equivalent yet (O8, O9) |
| 7 | Evaluation through decoded latent trajectories (§3.3.3, §4.1) | **CHANGED** | Decode predicted states with Dreamer's own image decoder |
| 8 | Quantitative evaluation; DreamerV3 and Director baselines (§3.3.3) | **KEPT** | Director becomes the closer comparison |
| 9 | Director as related work (§2.5.2) | **CHANGED** | "Trajectory goals vs state goals" no longer separates us |
| 10 | HER as related work (§2.5.3) | **CHANGED** | Now kinship rather than contrast |
| 11 | HAC as the template for the hierarchy (§2.4.1, §3.1) | **KEPT** | The fit is now closer |
| 12 | Trajectories as goals (§2.4.2, Co-Reyes 2018) | **UNDECIDED** | In tension with predicting states |
| 13 | Motivation: goals as explanations of past behaviour (Bennett 2021; §1, §2.5.4) | **UNDECIDED** | Needs reframing |
| 14 | The bootstrapping argument (§3.3.1, "Learning to accomplish goals") | **KEPT**, with a new open risk | O6, O8 |
| 15 | Research questions RQ1 and RQ2 (§3.2) | **KEPT** | |
| 16 | Domains: MiniGrid, then XLand-MiniGrid (§3.3.2) | **KEPT**; testbed **UNDECIDED** | O10 |
| 17 | Schedule (Table 4.1) | **CHANGED** | Item 3 dropped; item 4 is the current work |
| 18 | Risks (§4.1) | **KEPT** | The change is the proposal's own fallback, adopted early |

### Details

**1 · Encoder and decoder — DROPPED.** Dreamer's actor and critic are functions of s_t alone, so a compressed window of past states can't tell you anything about the future that s_t doesn't. The embedding space τ existed to give the TM a target space, and future states can play that role directly. §4.1 names "simpler trajectory representations and shorter prediction horizons" as the fallback if embeddings disappoint; this change takes that fallback early, on principled grounds. Worth saying so in the thesis.

**2 · TM equations — CHANGED.** In the proposal the TM already predicted the future: Eq. 3.2 trains f_n to match the encoder's embedding of the segment that *followed* z_0 in replay. So the direction of prediction and the hindsight-style supervision are unchanged, and the **target representation** is what changes, from embeddings τ̂ to states ŝ.

- **KEPT:** the top level sees only the current state, and each finer level conditions on the next-coarser prediction.
- **UNDECIDED:** whether finer levels also see s_t (the proposal: yes, via z_0); the levels and horizons (O1); distributional versus deterministic output; teacher forcing (O3).

**3 · Eq. 3.2 — DROPPED.** It is a KL to the encoder's posterior p_φ(·|Z), which no longer exists. The replacement loss is O3.

**4 · Action model — CHANGED.** Still derived from the finest level, but its input changes from τ̂_0 to ŝ_{t+1}. Using ŝ_{t+1} alone is probably under-determined; (s_t, ŝ_{t+1}) makes it inverse dynamics (O5). Index carefully: the action that produces s_{t+1} is `action[t]`. Actions aren't yet in the encoded data.

**5 · z vs s — CHANGED.** The proposal uses z_t for "the RSSM latent state", but in DreamerV3's notation z_t is only the stochastic part. Read literally, the proposal discards the recurrent state h_t, which in partially observable DoorKey holds whether the key is held and whether the door is open. Write s_t ≐ {h_t, z_t} everywhere, including Eq. 3.3's π_θ(·|z_t) → π_θ(·|s_t). Decided (S3).

**6 · Confidence-gated auxiliary loss — KEPT in form, mechanism UNDECIDED.** Eq. 3.3, L_actor = L_Dreamer + λ_aux·exp(−β·Var[τ̂_0])·CE(sg(p_a(·|τ̂_0)), π_θ(·|z_t)), remains the intended integration. It keeps a stop-gradient on the TM side, cross-entropy alignment, and confidence weighting.

- Var[τ̂_0] needs a new source: a distributional TM or an ensemble (O8).
- Where the loss applies, given the posterior-versus-prior shift, is O9.
- The new risk is O6: by default the loss pulls the actor toward its own past behaviour, and variance-based confidence can't tell good behaviour from bad.
- Out of scope until the TM works offline.

**7 · Decoded latent trajectories — CHANGED.** There is no trajectory decoder to inspect. Instead, push predicted states ŝ_{t+k} through **Dreamer's own observation decoder** (`self.dec` in the frozen checkpoint) to render what the TM expects the agent to see k steps ahead. That is arguably more direct than decoding an embedding. Details: the decoder was trained on sampled one-hot `stoch`, so sample predicted categoricals or check how soft probabilities behave before trusting the output. The output is the 64×64 egocentric render, not a full map. Needs a decoder hook like `latents()`.

**8 · Quantitative evaluation and baselines — KEPT.** Sample efficiency, return, success rate and task complexity against DreamerV3 and Director, plus transfer to held-out layouts and tasks (RQ2). Director matters more now (item 9).

**9 · Director — CHANGED.** The proposal separates HUGs from Director as "trajectory-based latent goals" versus Director's "state-based latent subgoals". **The new TM's goals are predicted future latent states, so that distinction is gone** and the argument has to be rebuilt. What still differs:

1. **Source.** Director's manager is an RL policy choosing goals to maximise reward. The TM is a supervised predictor of the agent's own recorded future.
2. **Use.** Director conditions a worker on the goal and rewards reaching it. HUGs keeps Dreamer's single actor and adds an auxiliary action-alignment loss.
3. **Structure.** Director has one goal level at a fixed interval. The TM has several levels, each conditioned on the coarser one.

How the thesis states the contribution is UNDECIDED (O11).

**10 · HER — CHANGED.** The proposal contrasts HER's achieved-state relabelling with learning "the structure of behaviour itself". The TM's targets are now achieved future states taken from replay — the state the agent actually reached k steps later — which is hindsight supervision in spirit. Remaining difference: HER relabels goals to train a goal-conditioned policy by RL, while the TM is a goal-free predictor of the agent's own future used as an auxiliary signal. Framing UNDECIDED.

**11 · HAC — KEPT, closer fit.** HAC's levels output desired future states for the level below. The new TM's levels output predicted future states conditioned on the level above — the same shape. The proposal's stated difference still holds: HAC's levels are RL-trained toward an externally given goal, while the TM's are supervised and have no external goal.

**12 · Trajectories as goals — UNDECIDED.** §2.4.2 argues that goals are better represented as trajectories than as single states. The new TM predicts states. One reconciliation is that the full stack ŝ_{t+K}, …, ŝ_{t+1} *is* a predicted trajectory (sparse if the spacing is coarse), but the argument needs rethinking either way.

**13 · Motivation — UNDECIDED, reframe.** Bennett's account — inferring intent from past paths — was made concrete by the autoencoder, which produced an explicit "explanation" of a segment of behaviour. Without it, the story becomes learning from one's own past behaviour to predict one's own future. That still fits the self-supervised spirit, but the "goals as explanations" wording overreaches.

**14 · Bootstrapping argument — KEPT, with risk.** "Reward-driven behaviour supplies trajectories; the TM learns their structure; the TM then regularises the actor" survives intact. What it doesn't address: confidence gating guards against an *unreliable* TM, not against a reliable TM describing *mediocre* behaviour (O6, O8).

**16 · Domains — KEPT; testbed UNDECIDED.**
- The implementation feeds **64×64 RGB renders** of the 7×7 egocentric view, not the symbolic 7×7×3 encoding §3.3.2 describes.
- Whether DoorKey-8x8 can carry a long-horizon claim is O10.
- XLand-MiniGrid isn't set up yet.

**17 · Schedule — CHANGED.**
- Item 3, "Build trajectory encoder" (17 Sep – 25 Oct 2026): dropped.
- Item 4, "Multi-timescale trajectory prediction" (planned 28 Oct – 6 Dec 2026): now the current work, and it can start early.
- Items 5–6 (integration, auxiliary loss): unchanged in intent.
- Phase 1 item 2: half done — the MiniGrid DoorKey baseline exists; XLand doesn't.

**18 · Risks — KEPT.** Integration risk, "accurate predictions that don't improve control", and compute risk all still apply. The mitigations still apply, including ensemble disagreement as an alternative confidence measure.

---

## 8. Working conventions and infrastructure

- **Cluster-only.** No local pip installs, venvs or JAX. The environment is a conda env under `/datasets/ndisler/HUGs/venv`, built by `cluster/setup_environment.sbatch`. The login node is for editing files and submitting jobs only; the cluster's MOTD says AI-agent and IDE connections to it get killed.
- **Noah runs cluster commands and pastes output.** Deliverables are code plus the exact `sbatch` command.
- **Paths.** Code locally at `/Users/noahdisler/Desktop/HUGs`; on the cluster at `/home-mscluster/ndisler/projects/HUGs`. Data and logs at `/datasets/ndisler/HUGs/{logdir,slurm_logs}`. Full workflow in `CLUSTER_WORKFLOW.md`.
- **Cycle.** `./sync_to_cluster.sh --once` on the Mac → `sbatch cluster/<job>.sbatch` on the cluster → `sacct -j <id> --format=JobID,JobName,State,Elapsed,ExitCode` and `cat /datasets/$USER/HUGs/slurm_logs/<name>-<id>.out`.
- **Jobs.** `train_bigbatch` (Dreamer training, args forwarded to `dreamerv3/main.py`), `check_latents` (GPU), `encode_states` (GPU), `episode_stats` (CPU, stampede), `view_results` (Scope viewer: SSH tunnel to port 8000).
- **Local tests:** `/opt/anaconda3/bin/python3 -m pytest tm/trajectory/tests/ -v`. Pure numpy, no cluster needed.
- **Style.** The repo uses 2-space indentation (Google style). Each file in `tm/trajectory/` opens with a docstring stating its PIPELINE POSITION and, for validation scripts, the RESULT from its last run.
