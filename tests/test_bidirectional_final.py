import unittest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from bidirectional_protocol_final import paired_group_table,split_groups,paired_episodes


class FinalProtocolTests(unittest.TestCase):
    def test_duplicate_aliases_propagate_across_both_resolutions(self):
        data={'c4':SimpleNamespace(ids=np.array(['1','2','3']),labels=np.array(['A']*3)),
              'c8':SimpleNamespace(ids=np.array(['1','3','4']),labels=np.array(['A']*3))}
        maps={'c4':pd.DataFrame({'source_id':['1','2','3','4'],'retained_source_id':['1','2','3','3']}),
              'c8':pd.DataFrame({'source_id':['1','2','3','4'],'retained_source_id':['1','1','3','4']})}
        inputs={'c4':np.array([[1,2],[2,3],[3,4]],np.float32),'c8':np.array([[4,5],[5,6],[6,7]],np.float32)}
        table,aliases,audit=paired_group_table(data,inputs,maps)
        self.assertEqual(len(table),2)
        mapping=dict(zip(aliases.source_id,aliases.group))
        self.assertEqual(mapping['1'],mapping['2']);self.assertEqual(mapping['3'],mapping['4'])

    def test_no_training_selection_query_overlap_and_nested_shots(self):
        labels=np.repeat(['A','B','C','D','E','F'],[61,200,299,494,495,497])
        task={'base':['A','B','C','D'],'novel_sessions':[['E'],['F']]}
        split=split_groups(labels,task,1)
        train,val=set(split['train']),set(split['validation'])
        self.assertFalse(train&val)
        query=set(sum(split['base_query'].values(),[])+sum(split['novel_query'].values(),[]))
        support=set(sum(split['novel_support'].values(),[]))
        self.assertFalse(query&(train|val|support))
        for ep in paired_episodes(labels,task,split,1,3):
            for pool in ep['novel_support_order'].values():self.assertEqual(len(pool),20)
            self.assertTrue(set(sum(ep['base_support'].values(),[]))<=train)
            self.assertEqual({len(q) for q in ep['query'].values()},{15})
            self.assertFalse(set(sum(ep['query'].values(),[]))&(set(sum(ep['base_support'].values(),[]))|set(sum(ep['novel_support_order'].values(),[]))))


if __name__=='__main__':unittest.main()
