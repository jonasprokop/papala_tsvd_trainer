"""Stage 3 of model evaluation (CPU): read-only rescoring of saved predictions.

Stage 1 (GPU) trains with the validation objective; stage 2 (GPU) loads the
selected checkpoints and saves predictions.npz; this stage computes every
paper protocol from that file and writes separate, versioned, never-overwritten
results. Protocols can change later without touching models or checkpoints.

Per-neuron scores come from `score_protocols`; this module applies each
paper's population aggregation:
  cadena2024_formula_match  NeuroVisFit aggregation: FEV > 0.15 (strict), NaN
                      correlations -> 0, mean; only neurons for which
                      cadena_formula_applicable (every test image repeated).
                      Matches the calculation, not the published population.
  cadena2024_tracked  same aggregation over every neuron, using the tracked
                      adaptation where images are missing or unrepeated
  brainscore2018      median raw r and median SB ceiling, reported side by side
  brainscore_current  per animal (median r)^2 / mean-of-split-median ceiling,
                      keeping neurons with Papale reliability > 0.3, then mean
                      over animals (10 split-halves at default settings).
                      Named `brainscore_current` only when a MEASURED Papale
                      reliability is supplied (npz key `papale_reliab_measured`,
                      e.g. released `reliab` for MUA); with the reconstructed
                      single-unit estimate it is `brainscore_current_adapted`. The 0.3 threshold on Papale's
                      `reliab` coordinate is verified; equivalence of our
                      estimate to that coordinate on sorted units is not, so
                      this number is labelled `brainscore_current_adapted`.
  bbscore_zwm         median of SB(r)/max(ceiling, 1e-6), including BBScore's
                      special handling of constant-response neurons
  attenuation         median r / sqrt(ceiling) (estimates r with noise-free response)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .evaluation_protocols import (BALANCED_DRAWS, BRAINSCORE_SPLITS,
                                   METRIC_LEVELS, score_protocols)

EV_THRESHOLD_CADENA = 0.15
RELIABILITY_THRESHOLD_BRAINSCORE = 0.3


def _stat(values, fn):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return (float(fn(v)) if len(v) else None), int(len(v))


def brainscore_split_ceiling(per_neuron_per_split):
    """Brain-Score neural_common.py: mean of split-wise neuroid medians."""
    values = np.asarray(per_neuron_per_split, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('Need finite [neurons, splits] ceiling matrix')
    return float(np.mean(np.median(values, axis=0)))


def _per_group(values, groups, fn, select=None):
    out = {}
    for g in np.unique(groups):
        sel = groups == g if select is None else (groups == g) & select
        v, n = _stat(np.asarray(values, float)[sel], fn)
        out[str(g)] = dict(value=v, n_neurons=n)
    return out


# Export-vs-recomputed criterion, calibrated for the tracked TVSD repeat structure.
# Both values are means of BALANCED_DRAWS random subsamples of the same trials,
# so on identical data their difference is Monte-Carlo noise with SD about
# sqrt(2) * SE, SE = per-neuron SD over draws / sqrt(draws). Per neuron
# z = (exported - recomputed) / (sqrt(2) * SE); under "same data" |z| has median
# ~0.67, with heavier-than-normal tails because SE comes from 10 draws.
# Calibration (sim, 300 ragged neurons x 10 row-order shuffles): same data gave
# median |z| <= 0.75 and at most 1.0% of neurons with |z| > 4; 5% / 10% of
# trials dropped gave 0.91 / 1.20 and 2.0% / 3.3%; added noise 1.81 and 26%;
# swapped neurons fail outright.
#   FAIL  median |z| > 1.5  or  fraction(|z| > 4) > 2%
#   WARN  median |z| > 1.0  or  fraction(|z| > 4) > 1%   (reported, not raised)
# Differences of a few percent of trials sit at the WARN/FAIL boundary.
EXPORT_Z_MEDIAN_MAX = 1.5
EXPORT_Z_MEDIAN_WARN = 1.0
EXPORT_Z_TAIL = 4.0
EXPORT_TAIL_FRACTION_MAX = 0.02
EXPORT_TAIL_FRACTION_WARN = 0.01


def export_consistency(exported, recomputed, se):
    exported = np.asarray(exported, float)
    recomputed = np.asarray(recomputed, float)
    if exported.ndim != 1 or exported.shape != recomputed.shape:
        raise ValueError('Expected matching per-neuron exported/recomputed reliability vectors')
    diff = exported - recomputed
    ok = np.isfinite(diff)
    missing_one_side = np.isfinite(exported) != np.isfinite(recomputed)
    missing_both = ~np.isfinite(exported) & ~np.isfinite(recomputed)
    out = dict(n_compared=int(ok.sum()), n_total=int(exported.size),
               n_missing_one_side=int(missing_one_side.sum()),
               n_missing_both=int(missing_both.sum()),
               median_abs_diff=float(np.median(np.abs(diff[ok]))) if ok.any() else None,
               max_abs_diff=float(np.max(np.abs(diff[ok]))) if ok.any() else None,
               criterion=dict(z_median_max=EXPORT_Z_MEDIAN_MAX, z_median_warn=EXPORT_Z_MEDIAN_WARN,
                              z_tail=EXPORT_Z_TAIL, tail_fraction_max=EXPORT_TAIL_FRACTION_MAX,
                              tail_fraction_warn=EXPORT_TAIL_FRACTION_WARN))
    if se is None:
        out.update(passed=False, reason='no Monte-Carlo SE in scores; rescore with current scorer')
        return out
    se = np.asarray(se, float)
    if se.shape != exported.shape:
        raise ValueError('Monte-Carlo SE must have one value per neuron')
    usable = ok & np.isfinite(se) & (se > 0)
    exact = ok & np.isfinite(se) & (se == 0)
    invalid_se = ok & ~(usable | exact)
    out['n_invalid_se'] = int(invalid_se.sum())
    if not ok.any() or missing_one_side.any() or invalid_se.any():
        out.update(passed=False, warning=False, n_z=int(usable.sum()),
                   reason=('no comparable neurons' if not ok.any() else
                           'different missing-reliability neurons' if missing_one_side.any() else
                           'missing or negative Monte-Carlo SE for comparable neurons'))
        return out
    # SE 0 means a deterministic pass: the two must then agree to rounding.
    exact_ok = bool(np.all(np.abs(diff[exact]) < 1e-9)) if exact.any() else True
    z = np.abs(diff[usable]) / (np.sqrt(2) * se[usable])
    med = float(np.median(z)) if len(z) else None
    tail = float(np.mean(z > EXPORT_Z_TAIL)) if len(z) else None
    passed = exact_ok and (not len(z) or (med <= EXPORT_Z_MEDIAN_MAX
                                          and tail <= EXPORT_TAIL_FRACTION_MAX))
    out.update(n_z=int(len(z)), median_abs_z=med, max_abs_z=float(z.max()) if len(z) else None,
               fraction_abs_z_above_tail=tail, n_deterministic=int(exact.sum()),
               deterministic_agree=exact_ok, passed=bool(passed),
               warning=bool(passed and len(z) and (med > EXPORT_Z_MEDIAN_WARN
                                                   or tail > EXPORT_TAIL_FRACTION_WARN)))
    return out


def population_accounting(scores, groups):
    width = len(groups)
    rep_imgs = np.asarray(scores['n_repeated_images'])
    out = dict(n_neurons=int(width),
               n_with_ceiling=int(np.isfinite(scores['split_half_sb_reliability']).sum()),
               n_formula_applicable=int(np.sum(scores['cadena_formula_applicable'])),
               median_repeated_images=float(np.median(rep_imgs)) if width else None,
               min_repeated_images=int(rep_imgs.min()) if width else None,
               median_observed_images=float(np.median(scores['n_cadena_images'])) if width else None,
               median_observations=float(np.median(scores['n_observations'])) if width else None,
               per_group={})
    for g in np.unique(groups):
        sel = groups == g
        out['per_group'][str(g)] = dict(
            n_neurons=int(sel.sum()),
            n_with_ceiling=int(np.isfinite(scores['split_half_sb_reliability'][sel]).sum()),
            n_formula_applicable=int(np.sum(np.asarray(scores['cadena_formula_applicable'])[sel])),
            median_repeated_images=float(np.median(rep_imgs[sel])))
    return out


def summarize_protocols(scores, groups=None, measured_reliab=None,
                        reconstructed_reliab=None):
    width = len(scores['raw_repeated_pearson'])
    groups = np.asarray(groups) if groups is not None else np.zeros(width, dtype=int)
    r = scores['raw_repeated_pearson']
    rel = scores['split_half_sb_reliability']

    def cadena(select):
        # NeuroVisFit: corr[fev > 0.15]; NaN -> 0; np.mean. FEVE: feve[fev > 0.15]; np.mean.
        with np.errstate(invalid='ignore'):
            keep = select & (scores['cadena_explainable_fraction'] > EV_THRESHOLD_CADENA)
        cc = np.nan_to_num(np.asarray(scores['cadena_corr_to_average'], float)[keep], nan=0.0)
        fe = np.asarray(scores['cadena_feve'], float)[keep]
        per_group = {}
        for g in np.unique(groups):
            k = keep & (groups == g)
            c = np.nan_to_num(np.asarray(scores['cadena_corr_to_average'], float)[k], nan=0.0)
            per_group[str(g)] = dict(cc_avg_mean=float(np.mean(c)) if len(c) else None,
                                     feve_mean=float(np.mean(np.asarray(scores['cadena_feve'], float)[k]))
                                     if k.any() else None, n_neurons=int(k.sum()))
        return dict(cc_avg_mean=float(np.mean(cc)) if len(cc) else None,
                    feve_mean=float(np.mean(fe)) if len(fe) else None,
                    n_neurons=int(keep.sum()), n_considered=int(select.sum()),
                    n_excluded_fev=int(select.sum() - keep.sum()), per_group=per_group,
                    n_nan_correlations_set_to_zero=int(np.isnan(
                        np.asarray(scores['cadena_corr_to_average'], float)[keep]).sum()))
    applicable = np.asarray(scores['cadena_formula_applicable'], bool)
    cadena_formula_match = cadena(applicable)
    cadena_tracked = cadena(np.ones(width, bool))

    raw, n_raw = _stat(r, np.median)
    ceil, _ = _stat(rel, np.median)

    if measured_reliab is not None and reconstructed_reliab is not None:
        raise ValueError('Cannot combine measured and reconstructed Papale reliability')
    if measured_reliab is not None:
        filter_reliab = np.asarray(measured_reliab, float)
        reliability_source = 'measured_papale_reliab'
    elif reconstructed_reliab is not None:
        filter_reliab = np.asarray(reconstructed_reliab, float)
        reliability_source = 'exported_tracked_units (papale_reliab_balanced_adapted)'
    else:
        filter_reliab = np.asarray(scores['papale_reliab_balanced_adapted'], float)
        reliability_source = 'reconstructed_single_unit (papale_reliab_balanced_adapted)'
    if filter_reliab.shape != (width,):
        raise ValueError('Reliability vector must have one value per neuron')
    export_check = None
    if reconstructed_reliab is not None:
        export_check = export_consistency(
            filter_reliab, scores['papale_reliab_balanced_adapted'],
            scores.get('papale_reliab_balanced_adapted_mc_se'))
        if not export_check['passed']:
            raise ValueError('Exported reliability disagrees with the NPZ responses beyond '
                             f'Monte-Carlo error: {export_check}')
    per_group = {}
    split_reliability = np.asarray(scores['split_half_sb_by_split'], dtype=float)
    if split_reliability.ndim != 2 or split_reliability.shape[0] != width:
        raise ValueError('Brain-Score requires per-neuron, per-split ceiling values')
    for g in np.unique(groups):
        sel = ((groups == g) & np.isfinite(r) & np.isfinite(rel)
               & (filter_reliab > RELIABILITY_THRESHOLD_BRAINSCORE))
        if sel.any():
            r_med = float(np.median(r[sel]))
            # Original Brain-Score neural_common.py:
            # mean across splits of the median across neuroids, NOT the
            # median of each neuroid's mean split-half reliability.
            c_med = brainscore_split_ceiling(split_reliability[sel])
            per_group[str(g)] = dict(raw_median_r=r_med, ceiling_median=c_med,
                                     ceiling_aggregation='mean_of_split_medians',
                                     n_ceiling_splits=int(split_reliability.shape[1]),
                                     ceiled=r_med ** 2 / c_med, n_neurons=int(sel.sum()))

    # BBScore's BaseMetric floors raw ceilings <= 1e-6 to positive 1e-6.
    # The per-neuron scorer also matches its constant-response convention.
    bb_all = np.asarray(scores['bbscore_noise_corrected'], dtype=float)
    bb, n_bb = _stat(bb_all, np.median)
    att, n_att = _stat(scores['pearson_noise_corrected'], np.median)

    measured_check = None
    if measured_reliab is not None:
        # Informational: Papale's all-pairs reliab vs our recomputed balanced
        # estimate on the same repeats (random pairing, 10 subsamples). Not a
        # gate: it is the real-data counterpart of verify_papale_reliab.py.
        rec = np.asarray(scores['papale_reliab_balanced_adapted'], float)
        both = np.isfinite(filter_reliab) & np.isfinite(rec)
        d = filter_reliab[both] - rec[both]
        measured_check = dict(n_compared=int(both.sum()),
                              median_diff=float(np.median(d)) if len(d) else None,
                              median_abs_diff=float(np.median(np.abs(d))) if len(d) else None,
                              corr=(float(np.corrcoef(filter_reliab[both], rec[both])[0, 1])
                                    if both.sum() > 2 and np.std(filter_reliab[both]) > 0
                                    and np.std(rec[both]) > 0 else None))
    rel1 = filter_reliab
    recomputed = np.asarray(scores['papale_reliab_balanced_adapted'], float)
    reliability = dict(
        # Recomputed from this NPZ (row order changes Monte-Carlo draws slightly).
        papale_reliab_balanced_adapted_median=_stat(recomputed, np.median)[0],
        # The vector actually used for the >0.3 filter and n_above.
        filter_reliability_source=reliability_source,
        export_vs_recomputed=export_check,
        measured_vs_recomputed=measured_check,
        filter_reliability_median=_stat(rel1, np.median)[0],
        papale_oracle_balanced_adapted_median=_stat(scores['papale_oracle_balanced_adapted'], np.median)[0],
        papale_balanced_k_median=float(np.median(scores['papale_balanced_k'])),
        papale_balanced_n_images_median=float(np.median(scores['papale_balanced_n_images'])),
        papale_oracle_30rep_model_estimate_median=_stat(
            scores['papale_oracle_30rep_model_estimate'], np.median)[0],
        single_trial_pairwise_estimate_median=_stat(
            scores['single_trial_pairwise_reliability_estimate'], np.median)[0],
        n_above={str(t): int(np.sum(rel1 > t)) for t in (0.15, 0.3, 0.4)},
        n_scored=int(np.isfinite(rel1).sum()),
        split_half_sb_median=_stat(rel, np.median)[0],
        papale_oracle_adapted_median=_stat(scores['papale_oracle_adapted'], np.median)[0])
    if 'pairwise_reliability_within_session' in scores:
        reliability['sessions'] = {
            key: _stat(scores[key], np.median)[0] for key in (
                'pairwise_reliability_within_session',
                'pairwise_reliability_between_sessions',
                'median_repeats_per_session_cell',
                'session_offset_sd_over_signal_sd')}
        reliability['sessions']['median_n_sessions'] = float(np.median(scores['n_sessions']))
    return dict(
        metric_levels=METRIC_LEVELS,
        reliability=reliability,
        population=population_accounting(scores, groups),
        cadena2024_formula_match=cadena_formula_match,
        cadena2024_tracked=cadena_tracked,
        brainscore2018=dict(raw_median_r=raw, ceiling_median=ceil, n_neurons=n_raw),
        **{('brainscore_current' if measured_reliab is not None
            else 'brainscore_current_adapted'): dict(
            reliability_source=reliability_source,
            n_excluded_reliability=int(np.sum(np.isfinite(r) & np.isfinite(rel)
                                              & ~(filter_reliab > RELIABILITY_THRESHOLD_BRAINSCORE))),
            ceiled=(float(np.mean([v['ceiled'] for v in per_group.values()]))
                    if per_group else None), per_group=per_group)},
        bbscore_zwm=dict(ceiled_median=bb, n_neurons=n_bb,
                         n_nonpositive_ceiling=int(np.sum(
                             np.asarray(scores['bbscore_split_half_sb_reliability']) <= 0)),
                         n_clipped_ceiling=int(np.sum(
                             np.asarray(scores['bbscore_split_half_sb_reliability']) <= 1e-6)),
                         per_group=_per_group(bb_all, groups, np.median)),
        attenuation=dict(median=att, n_neurons=n_att,
                         per_group=_per_group(scores['pearson_noise_corrected'], groups, np.median)),
    )


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _provenance(args):
    here = Path(__file__).resolve().parent
    out = dict(stage='protocol_evaluation_cpu',
               created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
               predictions=str(args.predictions.resolve()),
               predictions_sha256=_sha256(args.predictions),
               scorer_sha256={f: _sha256(here / f) for f in
                              ('evaluation_protocols.py', 'compare_evaluations.py',
                               'reliability_binding.py') if (here / f).is_file()},
               splits=args.splits,
               brainscore_ceiling_splits=min(BRAINSCORE_SPLITS, args.splits),
               papale_balanced_draws=BALANCED_DRAWS,
               min_repeated_images=args.min_repeated_images)
    if args.reliability_export_dir:
        source = json.loads((Path(args.reliability_export_dir).expanduser()
                             / 'reliability_summary.json').read_text())['source']
        out['reliability_export'] = dict(
            dir=str(Path(args.reliability_export_dir).expanduser().resolve()),
            dataset_version=source.get('dataset_version'),
            split_sha256=source.get('split_sha256'),
            export_splits=source.get('splits'))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path,
                        help='NPZ with prediction, response, mask, image_id[, groups, neuron_ids, session]')
    parser.add_argument('--output', required=True, type=Path,
                        help='New .npz path for per-neuron protocol scores')
    parser.add_argument('--min-repeated-images', type=int, default=20)
    parser.add_argument('--splits', type=int, default=50,
                        help='50 BBScore/attenuation draws by default; Brain-Score uses first 10; '
                             'Papale balanced reliability always uses 10 subsamples')
    parser.add_argument('--reliability-export-dir', type=Path,
                        help='Reconstructed tracked-unit reliability export (adapted, not MUA)')
    parser.add_argument('--measured-reliability', action='store_true',
                        help='Normalized-MUA cohort: bind Papale\'s released reliab from '
                             '--prepared-root/neuron_index.csv (yields brainscore_current)')
    parser.add_argument('--prepared-root', type=Path,
                        help='Required with --reliability-export-dir for frozen dataset provenance checks')
    parser.add_argument('--widths', help='JSON like {"F":804,"N":369}; columns in F|N order. '
                        'Also supplies neuron IDs F/0.. N/.. when the NPZ has none (Sinz '
                        'predictions.npz), so seeds match reliability_export.')
    args = parser.parse_args()
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz (numpy would append it and bypass the overwrite check)')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Output already exists; use a distinct versioned name')
    with np.load(args.predictions, allow_pickle=False) as arrays:
        fields = ('prediction', 'response', 'mask', 'image_id')
        missing = set(fields) - set(arrays.files)
        if missing:
            parser.error(f'Missing required arrays: {sorted(missing)}')
        ids = arrays['neuron_ids'] if 'neuron_ids' in arrays.files else None
        measured = (arrays['papale_reliab_measured']
                    if 'papale_reliab_measured' in arrays.files else None)
        reconstructed = (arrays['papale_reliab_exported_adapted']
                         if 'papale_reliab_exported_adapted' in arrays.files else None)
        if ids is None and args.widths:
            w = json.loads(args.widths)
            ids = np.array([f'{a}/{i}' for a in ('F', 'N') for i in range(int(w[a]))])
        if args.measured_reliability:
            if args.reliability_export_dir:
                parser.error('--measured-reliability and --reliability-export-dir are exclusive')
            if args.prepared_root is None or ids is None:
                parser.error('--measured-reliability needs --prepared-root and neuron IDs')
            from .reliability_binding import load_measured_reliability
            measured = load_measured_reliability(args.prepared_root, ids)
        if args.reliability_export_dir:
            if args.prepared_root is None or ids is None:
                parser.error('--reliability-export-dir needs --prepared-root and neuron IDs/--widths')
            from .reliability_binding import load_tracked_reliability
            external = load_tracked_reliability(args.reliability_export_dir, ids,
                                                args.prepared_root)
            if reconstructed is not None and not np.allclose(
                    reconstructed, external, equal_nan=True):
                parser.error('Existing exported reliability in NPZ disagrees with supplied export')
            reconstructed = external
        sessions = arrays['session'] if 'session' in arrays.files else None
        scores = score_protocols(*(arrays[key] for key in fields),
                                 min_repeated_images=args.min_repeated_images,
                                 splits=args.splits, neuron_ids=ids, sessions=sessions)
        groups = arrays['groups'] if 'groups' in arrays.files else None
    if groups is None and args.widths:
        widths = json.loads(args.widths)
        groups = np.concatenate([[a] * int(widths[a]) for a in ('F', 'N')])
    if groups is not None and len(groups) != len(scores['raw_repeated_pearson']):
        parser.error('groups length does not match the neuron count')
    # Summarise (and run the export consistency check) BEFORE writing anything,
    # so a failed check leaves no partial output that would block a rerun.
    summary = summarize_protocols(scores, groups, measured_reliab=measured,
                                  reconstructed_reliab=reconstructed)
    summary['provenance'] = _provenance(args)
    if reconstructed is not None:
        # Both estimates are kept: exported (used for the filter) and recomputed
        # (`papale_reliab_balanced_adapted`), plus its Monte-Carlo SE.
        scores['papale_reliab_exported_adapted'] = reconstructed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **scores)
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
