"""Standalone ZWM train entrypoint: no second Python process or Sinz CLI imports.

The caller must already be in the selected Python 3.10 environment. Candidate
config and all results are read/written using the existing project RunRegistry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sinz_model.orchestrator import RunRegistry, candidate_config_sha256
from .train import train_ridge_member


def run(sqlite_path: str, candidate_id: str, member: int = 0) -> str:
    registry = RunRegistry(sqlite_path)
    cfg = registry.get_candidate_config(candidate_id)
    if cfg.get('model', {}).get('family') != 'transformer_neural':
        raise ValueError('The selected candidate is not a transformer_neural model')
    if cfg.get('training', {}).get('strategy') != 'frozen_ridge':
        raise NotImplementedError('Only frozen_ridge is currently implemented for transformer_neural')
    # project.sqlite_path is execution-only: standalone PBS workers receive a
    # private scratch snapshot, never the central network-storage SQLite file.
    cfg['project']['sqlite_path'] = str(Path(sqlite_path).resolve())
    if candidate_config_sha256(cfg) != registry.get_candidate(candidate_id)['config_sha256']:
        raise ValueError('Execution override changed candidate identity')
    runs = registry.candidate_runs(candidate_id, member_index=member)
    if len(runs) > 1:
        raise RuntimeError('Multiple runs for a single deterministic ridge member')
    if runs and runs[0]['status'] == 'completed':
        return runs[0]['run_id']
    return train_ridge_member(cfg, member, candidate_id, evaluate_test=False,
                              resume_run_id=(runs[0]['run_id'] if runs else None))


def main() -> None:
    parser = argparse.ArgumentParser(description='Train one ZWM candidate in the selected Python runtime')
    parser.add_argument('--sqlite-path', required=True)
    parser.add_argument('--candidate-id', required=True)
    parser.add_argument('--member', type=int, default=0)
    args = parser.parse_args()
    run_id = run(args.sqlite_path, args.candidate_id, args.member)
    print(json.dumps({'run_id': run_id, 'runtime_python': sys.executable}), flush=True)


if __name__ == '__main__':
    main()
