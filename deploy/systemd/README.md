# systemd user units for the autopilot daemons

Vendored copies of the live units under `~/.config/systemd/user/`. The live
copies are the source of truth at runtime; these exist so a fresh-machine
reinstall doesn't silently lose two load-bearing fixes:

1. **`PATH` in `autopilot@.service`** — systemd user services get a minimal
   `PATH` with no nvm bin, so `claude` (at `~/.nvm/.../bin/claude`) resolved to
   "command not found" (exit 127) and every triage failed silently. The
   `Environment=PATH=...` line prepends the nvm node bin.

2. **Resource cage in `autopilot@experiment.service.d/override.conf`** — the
   experiment phase executes LLM-generated `recompute_*.py` on local hardware.
   There is no GPU path (torch/transformers/jax are banned imports), so the
   crash vector is RAM: ~2172 NPZ caches, the largest 157 MB each. A script
   that stacks all per-problem 7B caches would need ~78 GB and OOM the 23 GB
   WSL VM. The drop-in caps the experiment phase at `MemoryMax=12G`,
   `MemorySwapMax=0` (fast clean OOM-kill, no swap thrash), `CPUQuota=1400%`
   (14 of 28 cores), `Nice=10`. Verified enforced via cgroup v2 with the
   cpu/memory controllers delegated to the user manager.

   No `MemoryHigh`: with swap disabled, a monotonically-growing runaway
   livelocks in direct reclaim at `MemoryHigh` instead of dying. Legit scripts
   use <2 GB and never approach the ceiling, so we want a fast hard kill, not a
   throttle.

## Install on a fresh machine

```bash
mkdir -p ~/.config/systemd/user/autopilot@experiment.service.d
cp deploy/systemd/autopilot@.service ~/.config/systemd/user/
cp deploy/systemd/autopilot@experiment.service.d/override.conf \
   ~/.config/systemd/user/autopilot@experiment.service.d/
systemctl --user daemon-reload
systemctl --user enable --now autopilot@triage autopilot@scriptgen autopilot@experiment
loginctl enable-linger "$USER"   # survive logout

# Verify the cage is enforced (must NOT be infinity):
systemctl --user show autopilot@experiment -p MemoryMax,MemorySwapMax,CPUQuotaPerSecUSec
```

Requires cgroup v2 with `cpu memory pids` delegated to the user manager
(`cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/cgroup.controllers`).
