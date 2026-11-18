# TVSD configurations

- `tvsd_default.yaml`: Sinz ConvNeXt model, training and split defaults.
- `tvsd_zwm_ridge.yaml`: frozen ZWM features and masked ridge readout. Runs
  in Python 3.10; check the exact neuron cohort and upstream checkpoint first.
- `tvsd_optuna_space.yaml`: **Sinz SGD only**, not a ZWM search space.
- `tvsd_search_space.yaml`, `tvsd_screen_plan.yaml`, `tvsd_local_sweep.yaml`:
  local screening and search plans.

Use explicit host/container paths for a MetaCentrum deployment. Do not point a
PBS worker at the central SQLite file for writes. See
[`transformer_model/README.md`](../transformer_model/README.md).
