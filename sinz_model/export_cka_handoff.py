#!/usr/bin/env python3
"""Pack ranked Sinz V1 ensembles and a small *raw-pixel* image probe into one ZIP.

Run in the TRAINER container from /project.  The actual model exports use the
trainer's existing, validated export_bundle(), not ad-hoc checkpoint discovery.
No changes to trainer runs, data, DB or checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from sinz_model.data import (ImageStore, PreprocessConfig,
                                      transform_image_geometry_numpy)
from sinz_model.export_bundle import export_bundle


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def config_for(conn, candidate_id: str) -> dict:
    row = conn.execute('SELECT config_json FROM candidates WHERE candidate_id=?',
                       (candidate_id,)).fetchone()
    if row is None:
        raise ValueError(f'Candidate {candidate_id} missing from SQLite registry')
    return json.loads(row[0])


def probe_settings(cfg: dict) -> dict:
    d = cfg['data']
    pp = d['preprocessing']
    return dict(images_subdir=d['images_subdir'],
                filename_pattern=d['image_filename_pattern'],
                image_id_offset=int(d['image_id_offset']),
                crop=pp['crop'], subsample=int(pp['subsample']),
                scale=float(pp['scale']), expected_output_hw=pp['expected_output_hw'])


def export_images(cfg: dict, target: Path, count: int, seed: int,
                  data_root: Optional[str], images_dir: Optional[str]) -> dict:
    dc = cfg['data']
    folder = (Path(images_dir).expanduser().resolve() if images_dir else
              (Path(data_root or dc['root']) / dc['images_subdir']).expanduser().resolve())
    store = ImageStore(str(folder), dc['image_filename_pattern'], int(dc['image_id_offset']))
    file_ids = store.list_file_ids()
    if len(file_ids) < count:
        raise ValueError(f'{folder} has {len(file_ids)} images, need {count}')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(file_ids), count, replace=False)
    preprocess = PreprocessConfig.from_dict(dc['preprocessing'])
    images = []
    provenance = []
    for j in chosen:
        file_id = int(file_ids[int(j)])
        image_id = file_id - store.image_id_offset
        raw = store.load(image_id)
        arr = transform_image_geometry_numpy(raw, preprocess)[0]
        if arr.shape != (93, 93):
            raise ValueError(f'CKA expects 93x93, got {arr.shape} for {file_id}')
        if not np.all(np.isfinite(arr)) or arr.min() < 0 or arr.max() > 255:
            raise ValueError(f'Raw transformed image {file_id} is outside [0,255]')
        images.append(arr)
        provenance.append({'image_id': image_id, 'file_id': file_id,
                           'filename': store.path_for_id(image_id).name})
    stacked = np.stack(images).astype(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, images=stacked)
    return dict(source=str(folder), n_images=count, seed=seed,
                image_sha256=hashlib.sha256(stacked.tobytes()).hexdigest(),
                file_sha256=sha256(target),
                preprocessing_geometry=probe_settings(cfg),
                pixel_domain='raw 0..255; geometry only; NO z-score',
                sample=provenance)


def build(args):
    root = Path(args.output_root).expanduser().resolve()
    rank_path = (Path(args.ranking).expanduser().resolve() if args.ranking else
                 root / 'ensemble_ranking.json')
    db_path = root / 'results.sqlite'
    out = Path(args.out).expanduser().resolve()
    if not rank_path.is_file():
        raise FileNotFoundError(f'Ranking missing: {rank_path}')
    if not db_path.is_file():
        raise FileNotFoundError(f'Trainer registry missing: {db_path}')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing archive: {out}')
    ranking = json.loads(rank_path.read_text())
    selected = ranking['ranking'][:args.top]
    if not selected:
        raise ValueError('Ranking is empty')
    ids = [x['candidate_id'] for x in selected]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate candidate IDs in ranking')
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as con:
        cfgs = [config_for(con, cid) for cid in ids]
    pp0 = probe_settings(cfgs[0])
    if any(probe_settings(cfg) != pp0 for cfg in cfgs[1:]):
        raise ValueError('Candidate image preprocessing geometry differs across models')

    source_dir = (args.images_dir or str(Path(args.data_root or cfgs[0]['data']['root'])
                                     / cfgs[0]['data']['images_subdir']))
    print(f'Source registry: {db_path}')
    print(f'Source images:   {source_dir}')
    print(f'Destination:     {out}')
    for i, row in enumerate(selected, 1):
        print(f'  {i:02d}. {row["candidate_id"]}  test_corr={row["test_corr_mean"]:.6f}')
    if args.dry_run:
        print('DRY RUN: no files copied')
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the ZIP to prevent cross-filesystem moves, and cleanup on failure.
    with tempfile.TemporaryDirectory(prefix='.cka_export_', dir=str(out.parent)) as tmp:
        stage = Path(tmp) / 'payload'
        models = stage / 'models'
        models.mkdir(parents=True)
        probe = export_images(cfgs[0], stage / 'probe_images.npz', args.count,
                              args.seed, args.data_root, args.images_dir)
        print(f'Exported {probe["n_images"]} raw 93x93 images', flush=True)
        exported = []
        for i, row in enumerate(selected, 1):
            cid = row['candidate_id']
            dest = models / cid
            print(f'[{i}/{len(selected)}] Validating & copying candidate {cid}', flush=True)
            export_bundle(str(db_path), cid, str(dest))
            cp_files = sorted((dest / 'checkpoints').glob('member_*_best.pt'))
            if len(cp_files) != 5:
                raise RuntimeError(f'{cid}: expected five exported checkpoints, got {len(cp_files)}')
            exported.append({'candidate_id':cid, 'rank':i,
                             'test_corr_mean':row['test_corr_mean'],
                             'members':[{'name':p.name,'sha256':sha256(p),
                                         'bytes':p.stat().st_size} for p in cp_files]})
        (stage / 'ensemble_ranking.json').write_text(json.dumps({
            **ranking, 'ranking':selected}, indent=2)+'\n')
        candidates = root / 'candidates.csv'
        if candidates.is_file():
            (stage / 'candidates.csv').write_bytes(candidates.read_bytes())
        manifest = dict(format='sinz_v1_cka_handoff_v1',
                        generated_at_utc=datetime.now(timezone.utc).isoformat(),
                        ranking_source=str(rank_path), trainer_registry=str(db_path),
                        candidates=exported, images=probe)
        (stage / 'handoff_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (stage / 'README.txt').write_text(
            'CKA handoff: models/<candidate_uuid>/{manifest.json,model.py,checkpoints/},\n'
            'probe_images.npz (key: images, raw 93x93 pixels), ensemble_ranking.json,\n'
            'candidates.csv (if present), handoff_manifest.json.\n'
            'All input datasets/checkpoints were read-only during export.\n')
        tmpzip = Path(tmp) / 'transfer.zip'
        with zipfile.ZipFile(tmpzip, 'w', allowZip64=True) as z:
            for p in sorted(stage.rglob('*')):
                if p.is_file():
                    # Checkpoints and npz already use compressed formats.
                    z.write(p, p.relative_to(stage).as_posix(),
                            compress_type=zipfile.ZIP_STORED)
        os.replace(tmpzip,out)
    print(f'CREATED: {out} ({out.stat().st_size/1024**3:.2f} GiB)')
    print(f'ZIP SHA256: {sha256(out)}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',default='/project/search',
                   help='Trainer search output root containing results.sqlite and ensemble_ranking.json')
    p.add_argument('--ranking',help='Optional override for ensemble_ranking.json')
    p.add_argument('--out',default='/project/cka_handoff.zip',help='Single resulting ZIP')
    p.add_argument('--top',type=int,default=8,help='Number of top ranked ensembles')
    p.add_argument('--count',type=int,default=192,help='Number of natural images')
    p.add_argument('--seed',type=int,default=124)
    p.add_argument('--data-root',help='Override dataset root if moved since training')
    p.add_argument('--images-dir',help='Override exact image directory')
    p.add_argument('--dry-run',action='store_true')
    args = p.parse_args()
    if args.count < 8 or args.top < 1:
        p.error('Need at least 8 images and 1 ensemble')
    build(args)

if __name__=='__main__':
    main()
