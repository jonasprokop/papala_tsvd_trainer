"""Identity proposals require no class labels and cannot merge simultaneous units."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from track_units import run,vector_norm,ISI_N_BINS,aggregate,match_score

class TrackingTest(unittest.TestCase):
    def test_unlabelled_tracks(self):
        sessions=['macaqueF_TVSD_20240112_B1','macaqueF_TVSD_20240112_B2',
                  'macaqueF_TVSD_20240115_B1','macaqueF_TVSD_20240115_B2']
        table=pd.DataFrame([dict(unit_instance_id=f'{s}/A1/uid{j}',animal='F',
                                session=s,array=1,electrode_id=21,firing_rate_hz=9.8+.1*i)
                            for i,s in enumerate(sessions) for j in [0]])
        w=vector_norm(np.sin(np.linspace(0,4*np.pi,90)))
        wave=np.stack([w]*len(table))
        isi=np.full((len(table),ISI_N_BINS),1/ISI_N_BINS)
        amp=np.full(len(table),100.)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'joint'
            result=run(Path(td),table,wave,amp,isi,rate_weight=2.0,
                       min_score=1.3,min_margin=0.5,outdir=out,write_details=True)
            mapping=pd.read_csv(out/'candidate_joint_unit_membership.csv')
            quality=pd.read_csv(out/'candidate_joint_track_quality.csv')
            self.assertEqual(result['n_unit_instances'],4)
            self.assertEqual(mapping.candidate_joint_track_id.nunique(),1)
            self.assertEqual(int(quality.n_days.iloc[0]),2)
            self.assertNotIn('final_class',mapping)
            self.assertNotIn('katka_classes',quality)

    def test_matching_weights_change_candidate_scores(self):
        session1='macaqueF_TVSD_20240112_B1'
        session2='macaqueF_TVSD_20240112_B2'
        table=pd.DataFrame([
            dict(unit_instance_id='a',animal='F',session=session1,array=1,
                 electrode_id=1,firing_rate_hz=10.),
            dict(unit_instance_id='b',animal='F',session=session2,array=1,
                 electrode_id=1,firing_rate_hz=14.),
        ])
        w1=vector_norm(np.sin(np.linspace(0,5,90)))
        w2=vector_norm(np.sin(np.linspace(.15,5.15,90)))
        wave=np.stack([w1,w2])
        isi=np.full((2,ISI_N_BINS),1/ISI_N_BINS)
        a=aggregate([0],table,wave,np.array([100.,125.]),isi)
        b=aggregate([1],table,wave,np.array([100.,125.]),isi)
        low=match_score(a,b,cross_day=False,rate_weight=0.0,
                        waveform_weight=0.0,amplitude_weight=0.0)
        high=match_score(a,b,cross_day=False,rate_weight=3.0,
                         waveform_weight=2.0,amplitude_weight=.8)
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertLess(high['score'],low['score'])
        self.assertGreater(high['rate_penalty'],low['rate_penalty'])
        self.assertGreater(high['waveform_penalty'],low['waveform_penalty'])

    def test_single_recording_is_retained_without_pairwise_correlation(self):
        session='macaqueN_TVSD_20220111_B1'
        table=pd.DataFrame([dict(unit_instance_id=f'{session}/A1/u0',animal='N',
                                session=session,array=1,electrode_id=1,
                                firing_rate_hz=8.0)])
        wave=np.stack([vector_norm(np.sin(np.linspace(0,4*np.pi,90)))])
        isi=np.full((1,ISI_N_BINS),1/ISI_N_BINS)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'joint'
            result=run(Path(td),table,wave,np.array([100.]),isi,
                       rate_weight=2.0,min_score=1.3,min_margin=0.5,
                       outdir=out,write_details=True)
            q=pd.read_csv(out/'candidate_joint_track_quality.csv')
            m=pd.read_csv(out/'candidate_joint_unit_membership.csv')
            self.assertEqual(result['n_global_tracks'],1)
            self.assertEqual(len(m),1)
            self.assertEqual(int(q.n_sessions.iloc[0]),1)
            self.assertTrue(np.isnan(q.min_pairwise_waveform_r.iloc[0]))
            self.assertFalse(bool(q.review_flag.iloc[0]))

if __name__=='__main__':unittest.main()
