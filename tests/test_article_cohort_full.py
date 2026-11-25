"""End-to-end synthetic fixture exercising all the adapter paths and 3-model filtering."""
import json, tempfile, unittest, sys
from pathlib import Path
import h5py, numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import article_figures as af
from article_cohorts import CohortSelector

class FullPanels(unittest.TestCase):
    def test_all_stored_curve_families_and_three_models(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); paths={};n=6
            for name,scale in [('original',1.),('founder',1.12),('third',.97)]:
                file=td/(name+'.h5'); paths[name]=file
                with h5py.File(file,'w') as f:
                    g=f.create_group('size_tuning')
                    g.create_dataset('radii',data=np.geomspace(.05,3,8))
                    a=g.create_group('curves');b=g.create_group('results')
                    c=f.create_group('contrast_response')
                    c.create_dataset('center_contrasts',data=np.array([0,.07,.25,.5]))
                    c.create_dataset('surround_contrasts',data=np.array([0,.125,.5]))
                    cc=c.create_group('curves')
                    d=f.create_group('contrast_size_tuning')
                    d.create_dataset('radii',data=np.geomspace(.05,3,8))
                    d.create_dataset('contrasts',data=[.03,.1,.5])
                    dc=d.create_group('curves')
                    o=f.create_group('orientation_tuning')
                    x=np.linspace(-np.pi/2,np.pi/2,9)
                    o.create_dataset('ori_shifts',data=x)
                    oc=o.create_group('curves_center');os=o.create_group('curves_surround_only');om=o.create_group('curves_center_surround')
                    k=f.create_group('center_contrast_surround_suppression')
                    conf=k.create_group('config')
                    conf.create_dataset('center_contrasts_ccss',data=[0,.07,.25,.5])
                    conf.create_dataset('surround_contrasts_ccss',data=[0,.125,.5])
                    ks=k.create_group('results_same');ko=k.create_group('results_orthogonal')
                    bw=f.create_group('black_white_preference').create_group('results')
                    tex=f.create_group('texture_noise_response')
                    gt=tex.create_group('texture');gn=tex.create_group('noise')
                    gt.attrs['description']='-'.join(str(j) for j in range(1,16))
                    hall=f.create_group('modulator_params')
                    for i in range(n):
                        label='neuron_'+str(i)
                        a.create_dataset(label,data=np.stack([np.linspace(.4,1.0,8)*scale,np.linspace(.05,.3,8)*scale]))
                        ds=b.create_dataset(label,data=[1.,2.,3.,.1+.07*i,.3,.8]);ds.attrs['suppression_present']=i%2==0 or name=='founder';ds.attrs['saturation_present']=False
                        cc.create_dataset(label,data=scale*np.outer([1,.8,.6],[.1,.5,.9,1.1])*(1+i/20))
                        dc.create_dataset(label,data=scale*np.outer([.3,.6,.9],np.array([.2,.4,.8,1.1,.8,.6,.4,.3]))*(1+i/20))
                        w=np.cos(x)**2
                        oc.create_dataset(label,data=(w+.2)*scale)
                        os.create_dataset(label,data=(.4*w+.1)*scale)
                        om.create_dataset(label,data=scale*((w+.3)[:,None]*(w+.2)[None,:]))
                        mat=scale*np.outer([1,.75,.5],[.1,.5,.9,1.1])*(1+i/20)
                        ks.create_dataset(label,data=mat); ko.create_dataset(label,data=mat*np.array([1,1.12,1.22])[:,None])
                        bw.create_dataset(label,data=[1.,2.,.3])
                        gt.create_dataset(label,data=np.full((15,3),2.0+scale+i/10))
                        gn.create_dataset(label,data=np.full((15,3),1.0+scale+i/10))
                        rows=[]
                        for theta in [-np.pi/2, -np.pi/4, 0, np.pi/4]:
                            for phase in np.linspace(0,2*np.pi,8,endpoint=False):
                                rows.append([theta,0,phase,1+0.5*np.cos(phase)+.2*np.cos(2*theta)])
                        hall.create_dataset(label,data=rows)
            cav={'SI_histogram':{'bins':[[0,.2],[.2,.4],[.4,.6],[.6,.8],[.8,1.0]],'SI_proportions':[.1,.25,.3,.25,.1]}}
            fre={'v1_modulation_index_distribution':{'bin_centers_digitized':[-.6,-.3,0,.3,.6],
                    'proportion_of_cells':[.05,.1,.6,.2,.05]},
                 'texture_family_modulation':{'texture_family':list(range(1,16)),
                       'V1':{'modulation_index':[.12]*15,'errorbar_endpoint_low':[.1]*15,
                             'errorbar_endpoint_high':[.14]*15}}}
            hal={'sosi_distribution':{'bin_centers':[0,.25,.5,.75,1], 'proportion_of_cells':[.15,.2,.3,.25,.1]},
                 'preferred_second_order_orientation_distribution':{'wrapped_proportion_of_cells':[.2,.3,.35,.15]}}
            corrs={i:c for i,c in enumerate([.88,.8,.95,.78,.9,.25])}
            runs={k:af.core.read_run(str(p)) for k,p in paths.items()}
            selector=CohortSelector(list(paths),anchor='original',quality_run='founder',threshold=.75,correlations=corrs)
            out=td/'figures'
            rep=af.run_all(paths,out,selector=selector,run_data=runs,refs=(cav,fre,hal,None,None))
            print('PLOTS',len(rep['plots']),'SKIPPED',rep['skipped'])
            # Both one-dimensional and two-dimensional protocols, and 3 models
            expected=['cavanaugh_size_tuning_curves.png','cavanaugh_SI_histogram.png',
                      'cavanaugh_contrast_response.png','cavanaugh_contrast_size.png',
                      'cavanaugh_orientation_tuning.png','cavanaugh_orientation_matrix.png',
                      'cavanaugh_CCSS_figure5A.png','cavanaugh_CCSS_figure5B.png',
                      'cavanaugh_CCSS_figure5C.png','yeh_model_energy_preference.png',
                      'freeman_MI_distribution.png','freeman_family_MI.png',
                      'hallum_SOSI.png','hallum_relative_orientation.png']
            for filename in expected:
                self.assertTrue((out/filename).is_file(),filename)
            self.assertEqual(len(rep['skipped']),0)
            self.assertEqual(rep['cohort_counts']['Freeman family MI'],{name:5 for name in paths})

if __name__=='__main__':unittest.main()
