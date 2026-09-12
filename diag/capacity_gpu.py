"""Is the Q-Former too small? Train a FRESH one on a purely discriminative objective and see
what it can reach. Also sweep capacity down to show whether size is the binding constraint."""
import json, sys, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0,'/work/hdd/bfvh/li26/Astrobridge-Train-v6/src')
from pathlib import Path
from captioner.model.captioner import FusionStack
from captioner.data.dataset import ModalityCacheReader
ROOT=Path('/work/hdd/bfvh/li26/Astrobridge-Train-v6')
SP={'spectra':'fd1b0e0e80261800','lightcurve':'100c41666f995408'}
MAXT={'spectra':512,'lightcurve':243}; DIM={'spectra':768,'lightcurve':384}
DEV='cuda'; torch.manual_seed(0); np.random.seed(0)

def load(mod, labels):
    m=pd.read_parquet(ROOT/'outputs/manifest/manifest.parquet')
    rd=ModalityCacheReader(ROOT/'outputs/cache',mod,SP[mod])
    rows=[(o,str(labels.get(o))) for o in m.loc[(m['split']=='train')&m[f'has_{mod}'],'object_id']
          if o in rd.index.index and str(labels.get(o)) not in ('None','nan','?','other')]
    c=pd.Series([l for _,l in rows]).value_counts(); keep=set(c[c>=40].index)
    rows=[(o,l) for o,l in rows if l in keep][:900]
    cls=sorted(keep)
    X=torch.zeros(len(rows),MAXT[mod],DIM[mod]); M=torch.ones(len(rows),MAXT[mod],dtype=torch.bool)
    for i,(o,_) in enumerate(rows):
        a=torch.from_numpy(rd.get(o)); k=min(a.shape[0],MAXT[mod]); X[i,:k]=a[:k]; M[i,:k]=False
    y=torch.tensor([cls.index(l) for _,l in rows])
    return X.to(DEV),M.to(DEV),y.to(DEV),cls

def train(mod,X,M,y,cls,nq,ds,epochs=40):
    n=len(y); g=torch.Generator().manual_seed(0); perm=torch.randperm(n,generator=g)
    tr,te=perm[:int(.8*n)].to(DEV),perm[int(.8*n):].to(DEV)
    fs=FusionStack({mod:DIM[mod]},ds,4096,dict(n_queries=nq,d_model=ds,n_layers=3,n_heads=6,ffn_mult=4,dropout=0.1),2,0.1).to(DEV)
    head=nn.Linear(4096,len(cls)).to(DEV)
    opt=torch.optim.AdamW(list(fs.parameters())+list(head.parameters()),lr=3e-4,weight_decay=0.01)
    best=0
    for ep in range(epochs):
        fs.train(); idx=tr[torch.randperm(len(tr),device=DEV)]
        for s in range(0,len(idx),32):
            b=idx[s:s+32]
            loss=nn.functional.cross_entropy(head(fs({mod:{'tokens':X[b],'mask':M[b]}}).mean(1)),y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        fs.eval()
        with torch.no_grad():
            pr=torch.cat([head(fs({mod:{'tokens':X[te[s:s+64]],'mask':M[te[s:s+64]]}}).mean(1)).argmax(1) for s in range(0,len(te),64)])
            acc=(pr==y[te]).float().mean().item(); best=max(best,acc)
            if ep%10==9 or ep==0:
                P=fs({mod:{'tokens':X[te[:64]],'mask':M[te[:64]]}}).reshape(min(64,len(te)),-1)
                r=((P-P.mean(0)).norm(dim=1).mean()/P.mean(0).norm()).item()
                print(f'      ep{ep:>2} acc={acc:.3f} ||dev||/||mean||={r:.3f}',flush=True)
    return best

m=pd.read_parquet(ROOT/'outputs/manifest/manifest.parquet')
L={'spectra':json.load(open(ROOT/'diag/spec_labels.json')),
   'lightcurve':dict(zip(m['object_id'],m['class_label']))}
for mod in ['spectra','lightcurve']:
    X,M,y,cls=load(mod,L[mod])
    maj=max((y==i).float().mean().item() for i in range(len(cls)))
    print(f'\n### {mod}: n={len(y)} classes={cls} majority={maj:.3f}',flush=True)
    for nq,ds in [(64,384),(16,384),(64,128),(8,64)]:
        print(f'   -- n_queries={nq} d_shared={ds} ({"SHIPPED CONFIG" if (nq,ds)==(64,384) else "smaller"})',flush=True)
        b=train(mod,X,M,y,cls,nq,ds)
        print(f'      BEST test acc = {b:.3f}   (majority {maj:.3f})',flush=True)
