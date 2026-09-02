# MS Cluster Workflow

This project keeps source code in the cluster home filesystem and all large or
generated files in the datasets filesystem:

- Code: `/home-mscluster/ndisler/projects/HUGs`
- Results: `/datasets/ndisler/HUGs/logdir`
- Slurm output: `/datasets/ndisler/HUGs/slurm_logs`
- Python environment: `/datasets/ndisler/HUGs/venv`

The `logdir`, `slurm_logs`, and `venv` entries inside the remote project are
symlinks to those datasets paths. The sync script excludes them, so `rsync
--delete` cannot remove cluster results or the environment.

## One-time setup

From the local project directory:

```bash
cd ~/Desktop/HUGs
./sync_to_cluster.sh --once
ssh mscluster-login 'cd /home-mscluster/ndisler/projects/HUGs && sbatch cluster/setup_environment.sbatch'
```

The environment installation downloads the CUDA-enabled JAX dependencies and
can take several minutes. It runs as a Slurm job so that it survives SSH
disconnects. Its output is written to `slurm_logs/hugs_setup-<job-id>.out`, and
its package caches are also stored under `/datasets/ndisler/HUGs`.

## Starting a work session

Open three terminal tabs.

### Tab 1 — live code sync

```bash
cd ~/Desktop/HUGs
./sync_to_cluster.sh
```

The script performs an initial sync, then watches for changes. If `fswatch` is
not installed, it safely falls back to polling every three seconds. Stop it with
Ctrl+C.

To inspect changes without touching the cluster:

```bash
./sync_to_cluster.sh --once --dry-run
```

### Tab 2 — view results

Submit the viewer as a compute job (never run it on the login node):

```bash
ssh mscluster-login 'cd /home-mscluster/ndisler/projects/HUGs && sbatch --parsable cluster/view_results.sbatch'
ssh mscluster-login 'squeue -u ndisler -o "%.18i %.18j %.2t %.12M %N"'
```

Once `hugs_viewer` is running, note its compute node from the final column and
open a tunnel through the login node, replacing `<compute-node>`:

```bash
ssh -N -L 8000:<compute-node>:8000 mscluster-login
```

Open <http://localhost:8000>. This uses the Scope viewer already listed in the
project requirements. Stop the tunnel with Ctrl+C and cancel the viewer job with
`scancel <job-id>` when finished.

### Tab 3 — cluster terminal

```bash
ssh mscluster-login
cd ~/projects/HUGs
```

Submit the default small Crafter training job to the GPU partition:

```bash
sbatch cluster/train_bigbatch.sbatch
```

Pass Dreamer arguments after the script name for a different experiment:

```bash
sbatch cluster/train_bigbatch.sbatch --configs atari size12m --script train_eval
```

For MiniGrid:

```bash
sbatch cluster/train_bigbatch.sbatch --configs minigrid size1m --script train_eval
```

The `gridworld` config from the reference script is not present in this HUGs
checkout, so it is intentionally not the default.

## Monitoring jobs

```bash
squeue -u "$USER"
sacct -u "$USER" --starttime today --format=JobID,JobName,State,Elapsed,ExitCode
ls -lt /datasets/$USER/HUGs/slurm_logs | head
tail -f /datasets/$USER/HUGs/slurm_logs/<logfile>.out
```

Cancel a job only when needed:

```bash
scancel <job-id>
```

## Normal cycle

1. Edit locally; Tab 1 syncs source changes.
2. Submit work from Tab 3.
3. Monitor Slurm output and inspect results in Tab 2.
4. Commit milestones locally with Git.

Never run Python, environment setup, the results viewer, or training on the
login node. Use it only for file transfer, Slurm commands, and SSH tunneling.
