# Papale/Roelfsema–Sinz TVSD: development environment (phase 1)

**Use the GPU server over existing VS Code Remote-SSH.** The connection is encrypted SSH; this project does NOT create a public VS Code Tunnel, an SSH reverse tunnel, or any published host ports. If SSH or SSH forwarding is restricted by local admin policy, use only methods approved for the server.

This is an inspect-first scaffold: **no raw-data copying or trainer migration yet**. It retains the original Sinz environment, pins the Hugging Face ConvNeXtV2 checkpoint, reserves **host GPU 0**, and mounts the raw TVSD data **read-only**. Code, reports, staged data, and outputs live on the host and survive container recreation.

## Permissions and safety

- No `privileged`, `network_mode: host`, `ports`, Docker socket, SSH private-key, or secret mounts.
- Original raw data: host `${TVSD_SOURCE}` -> container `/source/tvsd` (read-only). Read permissions are still required on the host.
- Project root -> `/project` (writable). Working disk `${TVSD_WORKDIR}` -> `/project/data` (writable). *Do not point `TVSD_WORKDIR` at the original raw dataset.*
- Container runs as your numeric UID:GID rather than as root; Docker itself remains a powerful host privilege. Using `docker` normally requires pre-existing authorization; **do not ask for sudo or add yourself to the docker group just to make this work**.
- No automatic restart after host reboot. Container idles using `sleep infinity` and stays alive when you close a shell or disconnect.
- Access through existing SSH; VS Code's internal Remote-SSH communication may involve SSH forwarding. Explicitly publishing Jupyter/SSH ports is neither necessary nor configured. Automatic forwarding of application ports is disabled in this devcontainer.

## Setup on the GPU host

Extract this package to `/home/jprokop/papale_roelfsema_sinz_model_adaptation/` (or move its contents into your existing project). If you previously unpacked the older devcontainer patch, **remove or rename the old `docker-compose.override.yml`**, since Compose automatically applies it. Keep any manually modified files backed up.

```bash
cd /home/jprokop/papale_roelfsema_sinz_model_adaptation
cp .env.example .env        # if .env doesn't already exist; do not overwrite your settings
# Set TVSD_UID to `id -u`, TVSD_GID to `id -g` in .env
# Set TVSD_WORKDIR (default: ./data) to a directory you own; create it below
mkdir -p data/{raw,prepared,reports,cache,runs}
docker compose config       # inspect mounts and GPU mapping (no changes)
docker compose up -d --build tvsd
# Check running service; no reboot restart is configured
docker compose ps
docker compose exec tvsd python scripts/check_environment.py
```

If TVSD_WORKDIR is an **external absolute path**, create its subdirectories there instead of in `./data`. Do not share this workspace with the original GPU 1 trainer.

## VS Code Dev Containers — over existing SSH

1. Connect to GPU server with **Remote-SSH** using your authorized normal SSH login.
2. Open `/home/jprokop/papale_roelfsema_sinz_model_adaptation` as a remote folder.
3. Run **Dev Containers: Reopen in Container**. Select the existing Compose service; don't create a separate Docker-in-Docker instance.

No extra VS Code Tunnel account/service or new listening server port is needed. Remote-SSH may fail if the server administrator has disabled required forwarding; that is a site policy decision, not a reason to elevate privileges.

## Enter and inspect

```bash
docker compose exec tvsd bash   # exit closes this shell, container keeps running

docker compose exec tvsd python scripts/inspect_tvsd.py \
  --root /source/tvsd --max-depth 4 \
  --report /project/data/reports/tvsd_inventory.json
```

If `spikes_filtered` exists immediately under the TVSD root, inspect it next:

```bash
docker compose exec tvsd python scripts/inspect_tvsd.py \
  --root /source/tvsd/spikes_filtered --max-depth 4 \
  --report /project/data/reports/spikes_filtered_inventory.json
```

Review inventories before copying selected files. The staging utility `scripts/stage_selected.py` is dry-run by default. Later we will add the real TVSD neural-response adapter, then migrate the original Sinz trainer/search code.

To stop it: `docker compose stop tvsd`. To start again: `docker compose start tvsd`. To remove only the container: `docker compose down` (host data remain). No reboot auto-start.

## Notes

The Docker base is the previously supplied Sinz scaffold (`sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7`), with Python/PyTorch/Transformers pins. This project has not been built or GPU-tested on your CSNG host. If the GPU host cannot access that image or the `/CSNG/` data path, verify host access before changing anything.
