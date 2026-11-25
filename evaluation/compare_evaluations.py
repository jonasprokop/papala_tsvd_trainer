"""Read-only prediction rescoring; writes separate, never-overwritten results.

Per-neuron scores come from `score_protocols`; this module applies each
paper's population aggregation:
  cadena2024_formula_match  NeuroVisFit aggregation: FEV > 0.15 (strict), NaN
                      correlations -> 0, mean; only neurons for which
                      cadena_formula_applicable (every test image repeated).
                      Matches the calculation, not the published population.
  cadena2024_tracked  same aggregation over every neuron, using the tracked
                      adaptation where images are missing or unrepeated
  brainscore2018      median raw r and median SB ceiling, reported side by side
  brainscore_current  per animal (median r)^2 / median ceiling, keeping neurons
                      with Papale reliability > 0.3, then mean over animals.
                      Named `brainscore_current` only when a MEASURED Papale
                      reliability is supplied (npz key `papale_reliab_measured`,
                      e.g. released `reliab` for MUA); with the reconstructed
                      single-unit estimate it is `brainscore_current_adapted`. The 0.3 threshold on Papale's
                      `reliab` coordinate is verified; equivalence of our
                      estimate to that coordinate on sorted units is not, so
                      this number is labelled `brainscore_current_adapted`.
  bbscore_zwm         median of SB(r)/ceiling over all neurons with a finite value
  attenuation         median r / sqrt(ceiling) (estimates r with noise-free response)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation_protocols import METRIC_LEVELS, score_protocols

EV_THRESHOLD_CADENA = 0.15
RELIABILITY_THRESHOLD_BRAINSCORE = 0.3


def _stat(values, fn):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return (float(fn(v)) if len(v) else None), int(len(v))


def _per_group(values, groups, fn, select=None):
    out = {}
    for g in np.unique(groups):
        sel = groups == g if select is None else (groups == g) & select
        v, n = _stat(np.asarray(values, float)[sel], fn)
        out[str(g)] = dict(value=v, n_neurons=n)
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


def summarize_protocols(scores, groups=None, measured_reliab=None):
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

    if measured_reliab is not None:
        filter_reliab = np.asarray(measured_reliab, float)
        reliability_source = 'measured_papale_reliab'
    else:
        filter_reliab = np.asarray(scores['papale_reliab_balanced_adapted'], float)
        reliability_source = 'reconstructed_single_unit (papale_reliab_balanced_adapted)'
    per_group = {}
    for g in np.unique(groups):
        sel = ((groups == g) & np.isfinite(r) & np.isfinite(rel)
               & (filter_reliab > RELIABILITY_THRESHOLD_BRAINSCORE))
        if sel.any():
            r_med, c_med = float(np.median(r[sel])), float(np.median(rel[sel]))
            per_group[str(g)] = dict(raw_median_r=r_med, ceiling_median=c_med,
                                     ceiled=r_med ** 2 / c_med, n_neurons=int(sel.sum()))

    # BBScore keeps nonpositive ceilings; score_protocols leaves them NaN, so
    # recompute the ratio here for every neuron with a finite r and ceiling.
    with np.errstate(divide='ignore', invalid='ignore'):
        bb_all = (2 * r / (1 + r)) / rel
    bb, n_bb = _stat(bb_all, np.median)
    att, n_att = _stat(scores['pearson_noise_corrected'], np.median)

    rel1 = scores['papale_reliab_balanced_adapted']
    reliability = dict(
        papale_reliab_balanced_adapted_median=_stat(rel1, np.median)[0],
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
                         n_nonpositive_ceiling=int(np.sum(rel <= 0)),
                         per_group=_per_group(bb_all, groups, np.median)),
        attenuation=dict(median=att, n_neurons=n_att,
                         per_group=_per_group(scores['pearson_noise_corrected'], groups, np.median)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path,
                        help='NPZ with prediction, response, mask, image_id[, groups, neuron_ids, session]')
    parser.add_argument('--output', required=True, type=Path,
                        help='New .npz path for per-neuron protocol scores')
    parser.add_argument('--min-repeated-images', type=int, default=20)
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
        if ids is None and args.widths:
            w = json.loads(args.widths)
            ids = np.array([f'{a}/{i}' for a in ('F', 'N') for i in range(int(w[a]))])
        sessions = arrays['session'] if 'session' in arrays.files else None
        scores = score_protocols(*(arrays[key] for key in fields),
                                 min_repeated_images=args.min_repeated_images,
                                 neuron_ids=ids, sessions=sessions)
        groups = arrays['groups'] if 'groups' in arrays.files else None
    if groups is None and args.widths:
        widths = json.loads(args.widths)
        groups = np.concatenate([[a] * int(widths[a]) for a in ('F', 'N')])
    if groups is not None and len(groups) != len(scores['raw_repeated_pearson']):
        parser.error('groups length does not match the neuron count')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **scores)
    summary = summarize_protocols(scores, groups, measured_reliab=measured)
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
