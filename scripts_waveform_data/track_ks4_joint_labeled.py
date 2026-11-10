#!/usr/bin/env python3
"""Label-aware, DAY-FIRST chronic-unit tracking for the filtered TVSD KS4 data.

Signals: z-scored waveform shape, log firing rate, ISI-distribution Hellinger
similarity, raw waveform amplitude, and full rebuilt Katka labels (conservative
polarity veto and small within-polarity adjustment only). Hungarian one-to-one assignments with
an explicit unmatched option, competition margins, and same-session exclusion.

Stage 1: reconstruct tracks within one recording day.
Stage 2: compare those daily tracks across dates within each physical electrode.

Nothing is written back to the original inventory, Katka references or NPZs.
All output IDs are *proposals*, never verified biological identities.

Run (inside the TVSD Docker project):
    python scripts/track_ks4_joint_labeled.py
    python scripts/track_ks4_joint_labeled.py --rate-weights 1 2 3
    python scripts/track_ks4_joint_labeled.py --root /project/data/processed/ks4_v1

Requires numpy, pandas, scipy. Reads original ragged spike times from existing
session NPZs; never reopens NIX or reprocesses images.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path('/project/data/processed/ks4_v1')
# Rate-normalized ISIs: preserves firing-pattern shape without counting mean
# firing rate twice (FR is scored explicitly in log space).
ISI_EDGES_REL = np.geomspace(0.025, 20.0, 24)
ISI_N_BINS = len(ISI_EDGES_REL) + 1  # underflow + between edges + overflow
SESSION_RE = re.compile(r'macaque([FN])_TVSD_(\d{8})_B(\d+)')


def parse_session(s: str) -> tuple[str, str, int]:
    m = SESSION_RE.fullmatch(str(s))
    if not m:
        raise ValueError(f'Invalid TVSD recording ID: {s}')
    return m.group(1), m.group(2), int(m.group(3))


def vector_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    if x.ndim != 1 or not np.isfinite(x).all():
        return np.full_like(x, np.nan)
    x -= np.mean(x)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else np.full_like(x, np.nan)


def isi_fingerprint(times: np.ndarray) -> tuple[np.ndarray, int]:
    if len(times) < 202:
        return np.full(ISI_N_BINS, np.nan, dtype=np.float32), max(0, len(times)-1)
    intervals = np.diff(times)
    intervals = intervals[np.isfinite(intervals) & (intervals >= 0)]
    if len(intervals) < 200:
        return np.full(ISI_N_BINS, np.nan, dtype=np.float32), len(intervals)
    # Include both tails, rather than silently dropping long ISIs/refractory violations.
    positive = intervals[intervals > 0]
    if len(positive) < 200:
        return np.full(ISI_N_BINS, np.nan, dtype=np.float32), len(intervals)
    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0:
        return np.full(ISI_N_BINS, np.nan, dtype=np.float32), len(intervals)
    hist, _ = np.histogram(intervals / scale, bins=np.r_[-np.inf, ISI_EDGES_REL, np.inf])
    p = hist.astype(np.float64)
    p /= p.sum()
    return p.astype(np.float32), len(intervals)


def extract_features(root: Path, table: pd.DataFrame, cache: Path,
                     recompute: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = table['unit_instance_id'].to_numpy(dtype=str)
    if cache.is_file() and not recompute:
        with np.load(cache) as d:
            if not np.array_equal(d['unit_instance_ids'], ids):
                raise ValueError('Feature cache does not match unit inventory; use --recompute-features')
            return (d['waveforms'].astype(float), d['amplitudes'].astype(float),
                    d['isi_histograms'].astype(float), d['n_isi'].astype(int))
    features = {}
    for k, session in enumerate(sorted(table.session.unique(), key=parse_session), start=1):
        with np.load(root / 'sessions' / f'{session}.npz') as d:
            uids = d['unit_instance_ids'].astype(str)
            wf = d['waveforms_zscored']
            raw = d['waveforms_raw']
            sp = d['spike_times_s']
            offsets = d['spike_offsets']
            if len(uids) != len(wf) or len(uids) != len(raw) or len(offsets) != len(uids)+1:
                raise ValueError(f'{session}: malformed unit arrays/offsets')
            if offsets[0] != 0 or offsets[-1] != len(sp) or np.any(np.diff(offsets) < 0):
                raise ValueError(f'{session}: invalid ragged spike offsets')
            for i, uid in enumerate(uids):
                if uid in features:
                    raise ValueError(f'Unit appears in multiple NPZs: {uid}')
                shape = vector_norm(wf[i])
                amplitude = (float(np.ptp(raw[i])) if np.isfinite(raw[i]).all() else np.nan)
                isi, ni = isi_fingerprint(sp[int(offsets[i]):int(offsets[i+1])])
                features[uid] = (shape, amplitude, isi, ni)
        print(f'Features: {k}/{table.session.nunique()} {session}', flush=True)
    if set(features) != set(ids):
        raise ValueError('Unit inventory does not match NPZ features')
    a = [features[i] for i in ids]
    wave = np.stack([x[0] for x in a])
    ampl = np.asarray([x[1] for x in a], dtype=float)
    isi = np.stack([x[2] for x in a])
    n_isi = np.asarray([x[3] for x in a], dtype=int)
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Cache only compact features, not the huge spike time arrays.
    with cache.open('wb') as f:
        np.savez_compressed(f, unit_instance_ids=ids, waveforms=wave.astype('f4'),
                            amplitudes=ampl.astype('f4'),
                            isi_histograms=isi.astype('f4'), n_isi=n_isi)
    return wave, ampl, isi, n_isi


@dataclass
class Profile:
    members: list[int]
    # Entries refer to the original table row positions.
    wave: np.ndarray
    rate: float
    isi: np.ndarray
    amp: float
    last_day: str
    sessions: set[str]
    days: set[str]
    classes: tuple[str, ...]


def aggregate(members: list[int], table: pd.DataFrame, wave: np.ndarray,
              amp: np.ndarray, isi: np.ndarray) -> Profile:
    sub = table.iloc[members]
    wav = wave[members]
    good = np.isfinite(wav).all(axis=1)
    shape = vector_norm(np.median(wav[good], axis=0)) if good.any() else np.full(wave.shape[1], np.nan)
    rates = sub.firing_rate_hz.to_numpy(dtype=float)
    rates = rates[np.isfinite(rates) & (rates > 0)]
    rate = float(np.median(rates)) if len(rates) else np.nan
    isis = isi[members]
    valid = np.isfinite(isis).all(axis=1)
    median_isi = np.median(isis[valid], axis=0) if valid.any() else np.full(isi.shape[1], np.nan)
    if np.isfinite(median_isi).all():
        median_isi = median_isi / median_isi.sum()
    amplitudes = amp[members]
    amplitudes = amplitudes[np.isfinite(amplitudes) & (amplitudes > 0)]
    amplitude = float(np.median(amplitudes)) if len(amplitudes) else np.nan
    sessions = set(sub.session.astype(str))
    days = {parse_session(s)[1] for s in sessions}
    if 'final_class' not in sub or sub.final_class.isna().any():
        raise ValueError('This tracker requires fully labeled unit_instances_labeled.csv')
    classes = tuple(sorted(sub.final_class.astype(str).unique()))
    return Profile(members.copy(), shape, rate, median_isi, amplitude,
                   max(days), sessions, days, classes)


def class_penalty(classes_a: tuple[str, ...], classes_b: tuple[str, ...]) -> float | None:
    """Waveform-derived labels: weak extra evidence, never a new neuron ID.

    Opposite waveform POLARITY is a strong warning and blocks an automated match.
    Width and first-peak labels are derived from the same waveforms scored already,
    so they contribute only a deliberately small disagreement penalty.
    """
    left, right = set(classes_a), set(classes_b)
    if not left or not right:
        raise ValueError('Missing original Katka waveform class')
    if ('UP' in left and any(x.startswith('DOWN_') for x in right) or
        'UP' in right and any(x.startswith('DOWN_') for x in left)):
        return None
    if left & right:
        return 0.0
    def distance(a: str, b: str) -> float:
        if a == b: return 0.0
        if a == 'UP' or b == 'UP': return 0.0 if a == b else 10.0
        rank = {'narrow': 0, 'medium': 1, 'wide': 2}
        ap = a.split('_'); bp = b.split('_')
        w = abs(rank[ap[1]] - rank[bp[1]])
        return 0.20 * w + (0.10 if len(ap) == 3 and len(bp) == 3 and ap[2] != bp[2] else 0.0)
    return min(distance(a, b) for a in left for b in right)


def match_score(a: Profile, b: Profile, *, cross_day: bool, rate_weight: float):
    """Heuristic joint evidence, not a calibrated identity probability.

    Rate is on a log scale; waveform and amplitude differences permit more
    drift across days. ISI Hellinger distance is in [0,1]. Missing ISI merely
    removes the ISI contribution rather than rejecting the match.
    """
    r = float(np.dot(a.wave, b.wave)) if np.isfinite(a.wave).all() and np.isfinite(b.wave).all() else np.nan
    if not np.isfinite(r) or r < (0.90 if cross_day else 0.93):
        return None
    if not (np.isfinite(a.rate) and np.isfinite(b.rate) and a.rate > 0 and b.rate > 0):
        return None
    log_fr = abs(float(np.log(a.rate / b.rate)))
    if (log_fr > (1.6 if cross_day else 1.35)):
        return None  # reject >~5x change, even when waveforms look identical
    amp_d = (abs(float(np.log(a.amp/b.amp))) if np.isfinite(a.amp) and np.isfinite(b.amp)
             and a.amp > 0 and b.amp > 0 else np.nan)
    if np.isfinite(a.isi).all() and np.isfinite(b.isi).all():
        h = float(np.linalg.norm(np.sqrt(a.isi) - np.sqrt(b.isi)) / np.sqrt(2.0))
    else:
        h = np.nan
    cls_penalty = class_penalty(a.classes, b.classes)
    if cls_penalty is None:
        return None
    # A linearized distance score. Labels are waveform-derived and receive
    # only a small adjustment to avoid double-counting the waveform signal.
    wave_penalty = 1.6 * (1-r) / (0.045 if cross_day else 0.022)
    rate_penalty = rate_weight * log_fr / (0.78 if cross_day else 0.53)
    isi_penalty = 1.0 * h / 0.28 if np.isfinite(h) else 0.0
    amp_penalty = 0.35 * amp_d / (0.85 if cross_day else 0.60) if np.isfinite(amp_d) else 0.0
    score = 4.5 - wave_penalty - rate_penalty - isi_penalty - amp_penalty - cls_penalty
    return {'score': score, 'waveform_r': r, 'log_rate_ratio': log_fr,
            'isi_hellinger': h, 'log_amplitude_ratio': amp_d,
            'waveform_penalty': wave_penalty, 'rate_penalty': rate_penalty,
            'isi_penalty': isi_penalty, 'amplitude_penalty': amp_penalty,
            'class_penalty': cls_penalty, 'classes_a': ';'.join(a.classes),
            'classes_b': ';'.join(b.classes)}


def hungarian_match(old: list[Profile], new: list[Profile], *, cross_day: bool,
                    rate_weight: float, min_score: float, min_margin: float,
                    ):
    """One-to-one matches with explicit dummy choices, plus competition checks."""
    n, m = len(old), len(new)
    if not n or not m:
        return [], [], []
    score = np.full((n, m), -1e9, dtype=float)
    evidence = {}
    for i, a in enumerate(old):
        for j, b in enumerate(new):
            if a.sessions & b.sessions or (a.days & b.days and cross_day):
                continue
            e = match_score(a, b, cross_day=cross_day, rate_weight=rate_weight)
            if e is not None:
                score[i,j] = e['score']
                evidence[(i,j)] = e
    valid = score >= min_score
    candidates = np.where(valid, score, -1e9)
    # Each new observation can stay unmatched through its own dummy column.
    assign = np.concatenate([candidates.T, np.zeros((m, m))], axis=1)
    rr, cc = linear_sum_assignment(assign, maximize=True)
    matches, rejected, accepted = [], [], []
    for j, col in zip(rr, cc):
        if col >= n or not valid[col, j]:
            continue
        i = int(col)
        s = float(score[i,j])
        competitor_for_old = score[i, np.arange(m) != j]
        competitor_for_new = score[np.arange(n) != i, j]
        # An alternative matters if it clears the absolute acceptance threshold.
        alt_a = (float(np.max(competitor_for_old[competitor_for_old >= min_score]))
                 if np.any(competitor_for_old >= min_score) else -np.inf)
        alt_b = (float(np.max(competitor_for_new[competitor_for_new >= min_score]))
                 if np.any(competitor_for_new >= min_score) else -np.inf)
        margin_a = s-alt_a if np.isfinite(alt_a) else np.inf
        margin_b = s-alt_b if np.isfinite(alt_b) else np.inf
        ev = dict(evidence[(i,j)], old_slot=i, new_slot=int(j),
                  margin_old=float(margin_a), margin_new=float(margin_b))
        if min(margin_a, margin_b) < min_margin:
            ev['reason'] = 'ambiguous_competitor'
            rejected.append(ev)
        else:
            matches.append((i,int(j)))
            accepted.append(ev)
    # Only actual matches modify tracks; rejected candidates remain separate.
    return matches, accepted, rejected


def run(root: Path, table: pd.DataFrame, wave: np.ndarray, amp: np.ndarray, isi: np.ndarray,
        *, rate_weight: float, min_score: float, min_margin: float,
        outdir: Path, write_details: bool) -> dict:
    all_daily = []
    all_global = []
    evidence_rows = []
    rejected_rows = []
    global_quality = []
    day_quality = []
    count_site = 0
    lower_bound = 0
    for (animal, array, electrode), rows in table.groupby(['animal','array','electrode_id'],sort=True):
        count_site += 1
        idxs = rows.index.to_list()
        site = table.iloc[idxs]
        lower_bound += int(site.groupby('session').size().max())
        by_date = {}
        for idx in idxs:
            s = table.iloc[idx]['session']
            by_date.setdefault(parse_session(s)[1], {}).setdefault(s, []).append(idx)
        daily = []
        for date in sorted(by_date):
            tracks = []
            for session in sorted(by_date[date], key=parse_session):
                inds = by_date[date][session]
                current = [aggregate([k], table, wave, amp, isi) for k in inds]
                matches, yes, no = hungarian_match(tracks,current,cross_day=False,
                    rate_weight=rate_weight,min_score=min_score,min_margin=min_margin)
                for ev in yes:
                    ev.update(stage='within_day', animal=animal,array=array,
                              electrode_id=electrode,date=date,session=session)
                for ev in no:
                    ev.update(stage='within_day', animal=animal,array=array,
                              electrode_id=electrode,date=date,session=session)
                for ev in yes + no:
                    ev['old_unit_ids'] = ';'.join(table.iloc[tracks[ev['old_slot']].members].unit_instance_id.astype(str))
                    ev['new_unit_ids'] = ';'.join(table.iloc[current[ev['new_slot']].members].unit_instance_id.astype(str))
                evidence_rows.extend(yes)
                rejected_rows.extend(no)
                assigned = set()
                for i,j in matches:
                    tracks[i] = aggregate(tracks[i].members + current[j].members,table,wave,amp,isi)
                    assigned.add(j)
                tracks.extend(prof for j,prof in enumerate(current) if j not in assigned)
            for track_no,p in enumerate(tracks,1):
                did=f'{animal}/A{array}/E{electrode}/D{date}/T{track_no:03d}'
                daily.append((date,did,p))
                for idx in p.members:
                    all_daily.append((table.iloc[idx].unit_instance_id,did))
                day_quality.append(dict(day_track_id=did,animal=animal,array=array,
                                        electrode_id=electrode,date=date,
                                        n_recordings=len(p.members),
                                        min_pairwise_r=track_min_r(p,wave)))
        global_tracks=[]
        global_day_ids=[]
        for date in sorted(by_date):
            batch=[(did,p) for d,did,p in daily if d==date]
            new=[p for _,p in batch]
            matches,yes,no=hungarian_match(global_tracks,new,cross_day=True,
                rate_weight=rate_weight,min_score=min_score,min_margin=min_margin)
            for ev in yes:
                ev.update(stage='across_days',animal=animal,array=array,
                          electrode_id=electrode,date=date,session='')
            for ev in no:
                ev.update(stage='across_days',animal=animal,array=array,
                          electrode_id=electrode,date=date,session='')
            for ev in yes + no:
                ev['old_unit_ids'] = ';'.join(table.iloc[global_tracks[ev['old_slot']].members].unit_instance_id.astype(str))
                ev['new_unit_ids'] = ';'.join(table.iloc[new[ev['new_slot']].members].unit_instance_id.astype(str))
            evidence_rows.extend(yes)
            rejected_rows.extend(no)
            matched=set()
            for i,j in matches:
                global_tracks[i]=aggregate(global_tracks[i].members + new[j].members,
                                            table,wave,amp,isi)
                global_day_ids[i].append(batch[j][0])
                matched.add(j)
            for j,(did,p) in enumerate(batch):
                if j not in matched:
                    global_tracks.append(p)
                    global_day_ids.append([did])
        for tr_no,(p, day_ids) in enumerate(zip(global_tracks,global_day_ids),1):
            tid=f'{animal}/A{array}/E{electrode}/JT{tr_no:03d}'
            for idx in p.members:
                all_global.append((table.iloc[idx].unit_instance_id,tid))
            s=table.iloc[p.members]
            r=s.firing_rate_hz.to_numpy(dtype=float)
            r=r[np.isfinite(r)&(r>0)]
            classes=sorted(s.final_class.dropna().astype(str).unique()) if 'final_class' in s else []
            days=sorted({parse_session(x)[1] for x in s.session})
            global_quality.append(dict(candidate_joint_track_id=tid,
                 animal=animal,array=array,electrode_id=electrode,
                 n_unit_instances=len(p.members),n_days=len(days),
                 n_daily_tracks=len(day_ids), n_sessions=s.session.nunique(),
                 min_pairwise_waveform_r=track_min_r(p,wave),
                 firing_rate_max_min_ratio=(float(max(r)/min(r)) if len(r) else np.nan),
                 n_katka_classes=len(classes),katka_classes=';'.join(classes),
                 review_flag=(len(classes)>1 or (len(r)>1 and max(r)/min(r)>3) or
                              (len(p.members)>1 and track_min_r(p,wave)<0.93)),
                 n_katka_labeled=int(s.final_class.notna().sum()) if 'final_class' in s else 0))
    day=pd.DataFrame(all_daily,columns=['unit_instance_id','candidate_day_track_id'])
    glob=pd.DataFrame(all_global,columns=['unit_instance_id','candidate_joint_track_id'])
    if len(day)!=len(table) or len(glob)!=len(table) or not day.unit_instance_id.is_unique or not glob.unit_instance_id.is_unique:
        raise AssertionError('Every original unit must belong to exactly one daily and global track')
    mapping=table.merge(day,on='unit_instance_id',validate='one_to_one').merge(glob,on='unit_instance_id',validate='one_to_one')
    if mapping.groupby(['candidate_joint_track_id','session']).size().max()>1:
        raise AssertionError('Collision: two simultaneously recorded units in same proposed track')
    gq=pd.DataFrame(global_quality)
    dq=pd.DataFrame(day_quality)
    summary=dict(rate_weight=rate_weight,min_score=min_score,min_margin=min_margin,
         n_unit_instances=len(table),n_sites=count_site,lower_bound=lower_bound,
         n_daily_tracks=len(dq),n_global_tracks=len(gq),
         tracks_spanning_multiple_days=int((gq.n_days>1).sum()),
         unit_instances_in_repeat_tracks=int(gq.loc[gq.n_unit_instances>1,'n_unit_instances'].sum()),
         tracks_with_katka_class_disagreement=int((gq.n_katka_classes>1).sum()),
         tracks_with_rate_ratio_over_3=int((gq.firing_rate_max_min_ratio>3).sum()),
         tracks_flagged_for_review=int(gq.review_flag.sum()),
         n_ambiguous_pair_proposals=len(rejected_rows))
    if write_details:
        outdir.mkdir(parents=True,exist_ok=True)
        mapping.to_csv(outdir/'candidate_joint_unit_membership.csv',index=False)
        gq.to_csv(outdir/'candidate_joint_track_quality.csv',index=False)
        dq.to_csv(outdir/'candidate_day_track_quality.csv',index=False)
        pd.DataFrame(evidence_rows).to_csv(outdir/'accepted_pair_evidence.csv',index=False)
        pd.DataFrame(rejected_rows).to_csv(outdir/'ambiguous_pair_evidence.csv',index=False)
    return summary


def track_min_r(p: Profile,wave:np.ndarray)->float:
    if len(p.members)<2: return np.nan
    w=wave[p.members]
    if not np.isfinite(w).all():return np.nan
    rs=w@w.T
    ix,jx=np.triu_indices(len(w),1)
    return float(rs[ix,jx].min())


def negatives(table:pd.DataFrame,wave:np.ndarray,amp:np.ndarray,isi:np.ndarray,weight:float)->pd.DataFrame:
    out=[]
    for _,r in table.groupby(['animal','array','electrode_id','session'],sort=True):
        if len(r)<2:continue
        indexes=r.index.to_list()
        ps=[aggregate([idx],table,wave,amp,isi) for idx in indexes]
        for i in range(len(ps)):
            for j in range(i+1,len(ps)):
                ev=match_score(ps[i],ps[j],cross_day=False,rate_weight=weight)
                if ev is not None:
                    out.append(ev)
    return pd.DataFrame(out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--inventory',type=Path,default=None,help='Default: ROOT/unit_instances_labeled.csv')
    ap.add_argument('--rate-weights',nargs='+',type=float,default=[1.,2.,3.],
                    help='Sensitivity sweep for relative firing-rate influence')
    ap.add_argument('--proposal-rate-weight',type=float,default=2.)
    ap.add_argument('--min-score',type=float,default=1.3)
    ap.add_argument('--min-margin',type=float,default=0.5)
    ap.add_argument('--recompute-features',action='store_true')
    args=ap.parse_args()
    if any(x<=0 for x in args.rate_weights+[args.proposal_rate_weight]):
        ap.error('Firing-rate weights must be positive')
    path=args.inventory if args.inventory is not None else args.root/'unit_instances_labeled.csv'
    if not path.exists():raise FileNotFoundError(path)
    table=pd.read_csv(path)
    required={'unit_instance_id','animal','array','electrode_id','session','firing_rate_hz'}
    if required-set(table):raise ValueError(f'Missing columns {required-set(table)}')
    if table.unit_instance_id.duplicated().any():raise ValueError('Duplicate unit IDs')
    if table.final_class.isna().any():
        raise ValueError('Some Katka classes are missing; run attach_katka_rebuilt.py first')
    if len(table) != 11012:
        print(f'NOTE: {len(table)} original KS4 unit instances, not expected 11012')
    if table.longitudinal_neuron_id.notna().any():
        raise ValueError('Input already contains assigned longitudinal identities')
    table=table.reset_index(drop=True)
    wave,amp,isi,n_isi=extract_features(args.root,table,
            args.root/'identity_diagnostics'/'joint_unit_features.npz',args.recompute_features)
    outdir=args.root/'identity_diagnostics'/'joint_labeled'
    print('Computing LABEL-AWARE day-first identity proposals (read only)...',flush=True)
    sw=[]
    for weight in sorted(set(args.rate_weights+[args.proposal_rate_weight])):
        print(f'\n=== Rate weight {weight:g} ===',flush=True)
        summary=run(args.root,table,wave,amp,isi,rate_weight=weight,
                min_score=args.min_score,min_margin=args.min_margin,
                outdir=outdir,write_details=(weight==args.proposal_rate_weight))
        print(pd.Series(summary).to_string(),flush=True)
        sw.append(summary)
    outdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(sw).to_csv(outdir/'joint_rate_weight_sweep.csv',index=False)
    neg=negatives(table,wave,amp,isi,args.proposal_rate_weight)
    if len(neg):
        print('\nKnown-distinct simultaneous units passing the waveform/FR gates (conditional sample):')
        print(neg.score.describe(percentiles=[.5,.9,.95,.99]).to_string())
        print(f'{(neg.score>=args.min_score).mean():.1%} also pass raw joint-score threshold '
              '(they are NEVER eligible for real matches; not an error-rate estimate)')
        neg.to_csv(outdir/'known_distinct_unit_scores.csv',index=False)
    print(f'\nSaved diagnostics to {outdir}\nNo longitudinal neuron IDs assigned; no unit instances deleted.')

if __name__=='__main__':main()
