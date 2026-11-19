"""短CAM报告审计仅使用人工JSON对象，不导入任何推理运行时。"""
import copy
import unittest
from scripts.audit_short_cam_validation import audit


def records():
    rows = [{'type':'plan','device':'npu:7','tolerances':{'fp32':{'cam_cosine_min':0.9999,'cam_max_abs':0.001}}},
            {'type':'environment','device':'npu:7','torch_npu':'reported'}]
    for i, duration in enumerate([0.3,0.999,1.001]):
        rows.append({'type':'clip','clip_id':str(i),'sample_id':'source','duration_s':duration,
                     'frames':round(duration*16000),'sample_rate':16000,'pcm_sha256':'a'*64,
                     'source_sha256_declared':'b'*64,'source_sha256_verified':'b'*64 if i==0 else None})
    for i in range(3):
        rows.append({'type':'reference','model':'speech_campplus','clip_id':str(i),'output':{'values':[1.0]*192}})
        rows.append({'type':'candidate','model':'speech_campplus','compiler':'eager','precision':'fp32',
                     'clip_id':str(i),'phase':'steady','output':{'values':[1.0]*192},
                     'comparison':{'passed':True,'valid':True,'cosine':1.0,'max_abs':0.0}})
    rows += [{'type':'cluster','compiler':'eager','precision':'fp32','same_partition':True,
              'reference_labels':{str(i):'spk_0' for i in range(3)},'candidate_labels':{str(i):'spk_0' for i in range(3)}},
             {'type':'candidate_summary','model':'speech_campplus','precision':'fp32','compiler':'eager','passed':True},
             {'type':'complete','baseline_passed':True}]
    return rows


class ShortCamAuditTests(unittest.TestCase):
    def test_all_real_duration_entries_and_192d_partition_pass(self):
        self.assertTrue(audit(records())['passed'])

    def test_identical_empty_cluster_partitions_cannot_hide_short_cam_drop(self):
        rows=records(); cluster=next(row for row in rows if row['type']=='cluster')
        cluster['reference_labels']={}; cluster['candidate_labels']={}
        result=audit(rows)
        self.assertFalse(result['passed'])
        self.assertTrue(any('include all clips' in item for item in result['issues']))

    def test_wrong_dimension_nonfinite_scale_and_missing_candidate_are_rejected(self):
        for mode in ['dimension','nan','scale','missing']:
            with self.subTest(mode=mode):
                rows=records(); index=next(i for i,row in enumerate(rows) if row['type']=='candidate')
                if mode=='dimension': rows[index]['output']['values']=[1.0]*191
                if mode=='nan': rows[index]['output']['values'][0]=float('nan')
                if mode=='scale': rows[index]['comparison']['max_abs']=0.1
                if mode=='missing': rows.pop(index)
                self.assertFalse(audit(rows)['passed'])

    def test_wrong_duration_unverified_source_and_no_complete_are_rejected(self):
        for mode in ['duration','hash','complete']:
            with self.subTest(mode=mode):
                rows=records()
                if mode=='duration': next(row for row in rows if row['type']=='clip')['duration_s']=0.299
                if mode=='hash':
                    for row in rows:
                        if row['type']=='clip': row['source_sha256_verified']=None
                if mode=='complete': rows.pop()
                self.assertFalse(audit(rows)['passed'])

    def test_partition_compares_membership_not_label_spelling(self):
        rows=records(); cluster=next(row for row in rows if row['type']=='cluster')
        cluster['candidate_labels']={str(i):'renamed_cluster' for i in range(3)}
        self.assertTrue(audit(rows)['passed'])
        cluster['candidate_labels']['0']='different'
        self.assertFalse(audit(rows)['passed'])
