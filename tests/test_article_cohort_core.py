import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from article_cohorts import CohortSelector, load_test_correlations
import article_figures as af


class CohortTests(unittest.TestCase):
    def test_quality_manifest_alignment_and_all_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            paths={}
            n=8
            for name in ['original','founder']:
                path=base/(name+'.h5'); paths[name]=path
                with h5py.File(path,'w') as f:
                    size=f.create_group('size_tuning')
                    size.create_dataset('radii',data=np.linspace(.08,3.2,9))
                    sg=size.create_group('curves')
                    sr=size.create_group('results')
                    yw=f.create_group('black_white_preference').create_group('results')
                    tex=f.create_group('texture_noise_response')
                    gt=tex.create_group('texture'); gn=tex.create_group('noise')
                    gt.attrs['description']='-'.join(str(j) for j in range(1,16))
                    for nid in range(n):
                        mult=1.1 if name=='founder' else 1.0
                        sg.create_dataset('neuron_'+str(nid),data=np.stack([
                            np.array([.1,.4,.8,1,1.2,1.0,.8,.6,.4])*mult,
                            np.array([.05,.06,.12,.17,.22,.25,.24,.22,.20])*mult]))
                        ds=sr.create_dataset('neuron_'+str(nid),data=[1.,2.,3.,.10+.05*nid,.3,.8])
                        ds.attrs['suppression_present'] = (nid%2==0) if name=='original' else True
                        ds.attrs['saturation_present'] = False
                        yw.create_dataset('neuron_'+str(nid),data=[.1,.2,.2])
                        a=np.full((15,4),1.0+0.1*nid)
                        b=np.full((15,4),.8+0.1*nid)
                        if nid==7:  # dropped row must not misalign family IDs
                            a[:]=np.nan
                        gt.create_dataset('neuron_'+str(nid),data=a)
                        gn.create_dataset('neuron_'+str(nid),data=b)
            corr_vals=[.93,.88,.78,.55,.95,.40,.60,.35]
            mf=base/'bundle.json';mf.write_text(json.dumps({'model':{'n_neurons':8,'correlations':corr_vals}}))
            runs={name:af.core.read_run(str(p)) for name,p in paths.items()}
            corrs, provenance=load_test_correlations(mf,runs['founder'])
            self.assertEqual({i for i,c in corrs.items() if c>=.75}, {0,1,2,4})
            self.assertEqual(provenance['n_neurons'],8)
            caval={'SI_histogram':{'bins':np.array([[0.,.25],[.25,.5],[.5,.75],[.75,1.]]),
                                     'SI_proportions':[.1,.3,.3,.3]}}
            fre={'v1_modulation_index_distribution':{'bin_centers_digitized':[-.6,-.3,0,.3,.6],
                                                       'proportion_of_cells':[.05,.1,.6,.2,.05]},
                 'texture_family_modulation':{'texture_family':list(range(1,16)),
                       'V1':{'modulation_index':[.12]*15,
                             'errorbar_endpoint_low':[.1]*15,
                             'errorbar_endpoint_high':[.14]*15}}}
            refs=(caval,fre,None,None,None)
            for slug,anchor,threshold,expected in [
                ('all_neurons',None,None,{'original':4,'founder':8}),
                ('anchor_original','original',None,{'original':4,'founder':4}),
                ('corr_ge_0p75',None,.75,{'original':3,'founder':3}),
                ('anchor_original_corr_ge_0p75','original',.75,{'original':3,'founder':3}),
            ]:
                selected=CohortSelector(list(paths),anchor=anchor, quality_run='founder',
                                        threshold=threshold, correlations=corrs)
                dest=base/slug
                rep=af.run_all(paths,dest,selector=selected,run_data=runs,refs=refs)
                self.assertEqual(rep['cohort_counts']['Cavanaugh SI histogram'],expected)
                self.assertTrue((dest/'cavanaugh_SI_histogram.png').is_file())
                self.assertTrue((dest/'cavanaugh_size_tuning_curves.png').is_file())
                self.assertTrue((dest/'freeman_family_MI.png').is_file())
                self.assertTrue((dest/'cohort_membership.csv').is_file())
                if threshold is not None:
                    self.assertEqual(rep['cohort_counts']['Freeman family MI'],{'original':4,'founder':4})
            # Exercise exactly the user-facing four-variant CLI, including
            # a single --corr-manifest entry and Python Path handling.
            result=subprocess.run([sys.executable,
                str(Path(__file__).resolve().parents[1]/'article_figures.py'),
                '--run','original='+str(paths['original']),
                '--run','founder='+str(paths['founder']),
                '--corr-manifest','founder='+str(mf),
                '--quality-run','founder',
                '--anchor-run','original',
                '--corr-threshold','0.75',
                '--make-variants', '--no-experiment',
                '--out',str(base/'cli_run')],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            self.assertTrue((base/'cli_run'/'VARIANT_INDEX.json').is_file())
            self.assertTrue((base/'cli_run'/'COHORT_COUNTS.csv').is_file())
            self.assertEqual(len(json.loads((base/'cli_run'/'VARIANT_INDEX.json').read_text())),4)

            # malformed, ambiguous positional vector must fail *before* plots
            bad=base/'bad.json';bad.write_text(json.dumps({'model':{'correlations':[.8,.9,.6]}}))
            with self.assertRaises(ValueError):
                load_test_correlations(bad,runs['founder'])


if __name__ == '__main__':
    unittest.main()
