"""Record/duplicate-group-safe paired C4/C8 protocol; not specimen provenance."""
import hashlib
import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self,values):
        self.parent={str(x):str(x) for x in values}

    def find(self,x):
        x=str(x)
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]]
            x=self.parent[x]
        return x

    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:
            low,high=sorted([a,b],key=int)
            self.parent[high]=low


def paired_group_table(data,inputs,mappings):
    ids=set(mappings['c4'].source_id)|set(mappings['c8'].source_id)
    uf=UnionFind(ids)
    for frame in mappings.values():
        for a,b in zip(frame.source_id,frame.retained_source_id):uf.union(a,b)
    initial=len({uf.find(s) for s in ids})
    # Exact equivalence after the locked row-local preprocessing is also blocked.
    fingerprints={}
    exact_edges=0
    for source in ['c4','c8']:
        for source_id,row in zip(data[source].ids,inputs[source]):
            canonical=np.asarray(row,dtype='<f4').copy();canonical[canonical==0]=0
            key=hashlib.sha256(canonical.tobytes()).hexdigest()
            if key in fingerprints:
                before=uf.find(source_id)!=uf.find(fingerprints[key])
                uf.union(source_id,fingerprints[key]);exact_edges+=int(before)
            else:fingerprints[key]=source_id
    groups={}
    for source in ['c4','c8']:
        for i,s in enumerate(data[source].ids):
            root=uf.find(s)
            groups.setdefault(root,{'c4':[],'c8':[]})[source].append(i)
    members={root:[] for root in groups}
    for s in ids:members[uf.find(s)].append(s)
    records=[]
    for root,g in sorted(groups.items(),key=lambda p:int(p[0])):
        if not g['c4'] or not g['c8']:raise ValueError('A conservative record group lacks a resolution')
        labels={str(data[source].labels[i]) for source in ['c4','c8'] for i in g[source]}
        if len(labels)!=1:raise ValueError('Duplicate/ID component crosses class labels')
        indices={source:{str(data[source].ids[i]):i for i in g[source]} for source in ['c4','c8']}
        shared=set(indices['c4'])&set(indices['c8'])
        chosen={source:min(shared or set(indices[source]),key=int) for source in ['c4','c8']}
        records.append(dict(group='g'+root,label=next(iter(labels)),c4_index=indices['c4'][chosen['c4']],
            c8_index=indices['c8'][chosen['c8']],c4_id=chosen['c4'],c8_id=chosen['c8'],
            same_representative_id=chosen['c4']==chosen['c8'],original_ID_count=len(members[root]),
            retained_c4_count=len(g['c4']),retained_c8_count=len(g['c8'])))
    aliases=pd.DataFrame([dict(source_id=s,group='g'+uf.find(s)) for s in sorted(ids,key=int)])
    return pd.DataFrame(records),aliases,dict(alias_components_before_processed_hash=initial,
        exact_processed_additional_merges=exact_edges,paired_groups=len(records),
        same_representative_ID_groups=sum(r['same_representative_id'] for r in records),
        grouping_status='conservative shared-record-ID + exact native duplicate aliases + exact processed equality; specimen identity unverified')


def split_groups(labels,task,seed):
    roles=np.full(len(labels),'unused',object)
    train,val,base_query,novel_support,novel_query=[],[],{},{},{}
    for ci,c in enumerate(sorted(set(labels))):
        rng=np.random.default_rng(26000+1000*seed+ci)
        order=rng.permutation(np.flatnonzero(labels==c))
        if c in task['base']:
            nq=max(15,int(.20*len(order)));nv=max(5,int(.15*len(order)))
            if len(order)-nq-nv<20:raise ValueError('Insufficient base groups for train/val/query and 20 references')
            q,v,t=order[:nq],order[nq:nq+nv],order[nq+nv:]
            roles[q],roles[v],roles[t]='base_query','base_validation','base_train'
            base_query[c]=q.tolist();train.extend(t);val.extend(v)
        else:
            n=len(order)//2;s,q=order[:n],order[n:]
            if len(s)<20 or len(q)<15:raise ValueError('Insufficient novel support/query groups')
            novel_support[c],novel_query[c]=s.tolist(),q.tolist()
            roles[s],roles[q]='novel_support_pool','novel_query_pool'
    return dict(train=sorted(map(int,train)),validation=sorted(map(int,val)),roles=roles.tolist(),
        base_query=base_query,novel_support=novel_support,novel_query=novel_query)


def paired_episodes(labels,task,split,seed,repeats):
    base={}
    rng=np.random.default_rng(36000+seed)
    train=np.asarray(split['train'])
    for c in sorted(task['base']):base[c]=rng.choice(train[labels[train]==c],20,replace=False).tolist()
    episodes=[]
    for r in range(repeats):
        rng=np.random.default_rng(46000+1000*seed+r)
        query={c:rng.choice(pool,15,replace=False).tolist() for c,pool in sorted({**split['base_query'],**split['novel_query']}.items())}
        novel={c:rng.permutation(pool)[:20].tolist() for c,pool in sorted(split['novel_support'].items())}
        episodes.append(dict(repeat=r,base_support=base,novel_support_order=novel,query=query))
    return episodes
