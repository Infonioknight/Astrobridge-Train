"""Does the cached IMAGE embedding encode what the pixels actually show?

Same test that exposed the spectra bug, applied to the image tier: compute properties straight
off the raw pixels that any working image encoder must capture, then check whether AION's cached
embedding predicts them. If nine hand-computed numbers beat a 768-d foundation-model embedding,
the encoder is not seeing the image.
"""
import os, sys
for v in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[v]='8'
import numpy as np, pandas as pd
sys.path.insert(0,'/work/hdd/bfvh/li26/Astrobridge-Train-v6/src')
from pathlib import Path
from captioner.data.dataset import ModalityCacheReader
from captioner.data.image_dataset import load_image_flux_pixels
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.extmath import randomized_svd

ROOT=Path('/work/hdd/bfvh/li26/Astrobridge-Train-v6')
m=pd.read_parquet(ROOT/'outputs/manifest/manifest.parquet')
pix=load_image_flux_pixels('gapatron/astrobridge-image-captions',
                           revision='d9fadc12fa3e2061b5c397b52f0e39bde10a2ec0',
                           cache_dir=Path('/work/hdd/bfvh/li26/hf-cache'))
print('pixel objects:',len(pix),flush=True)
rd=ModalityCacheReader(ROOT/'outputs/cache','image','43629215d240cc71')
idmap=dict(zip(m['object_id'],m.get('object_id_legacy',m['object_id'])))

def canon(b): return str(b).split('-')[-1].lower()
rows=[]; E=[]
for oid in m.loc[m['has_image'],'object_id']:
    if oid not in rd.index.index: continue
    pid=idmap.get(oid,oid)
    if pid not in pix: continue
    bands={canon(e['band']):np.stack([np.asarray(r,np.float32) for r in e['flux']]) for e in pix[pid]}
    if not {'g','r','z'} <= set(bands): continue
    g,r,z=bands['g'],bands['r'],bands['z']
    if not (np.isfinite(g).all() and np.isfinite(r).all() and np.isfinite(z).all()): continue
    H,W=g.shape; c=(H//2,W//2); h=48
    cut=lambda a: a[max(0,c[0]-h):c[0]+h, max(0,c[1]-h):c[1]+h]
    g,r,z=cut(g),cut(r),cut(z)
    tot=lambda a: float(np.clip(a,0,None).sum())+1e-6
    # central concentration: flux within 5px / flux within 48px
    inner=lambda a: float(np.clip(a[a.shape[0]//2-5:a.shape[0]//2+5, a.shape[1]//2-5:a.shape[1]//2+5],0,None).sum())
    feats=dict(
        log_flux_r=np.log10(tot(r)),
        gr_color=np.log10(tot(g)/tot(r)),
        rz_color=np.log10(tot(r)/tot(z)),
        concentration=inner(r)/tot(r),
        peak=float(np.log10(max(r.max(),1e-6)+1e-6)),
    )
    rows.append(feats); E.append(rd.get(oid).mean(0))
    if len(rows)>=900: break
df=pd.DataFrame(rows); E=np.stack(E)
print(f'{len(df)} objects with pixels + cached embedding; emb {E.shape}',flush=True)
Xc=E-E.mean(0,keepdims=True); U,S,_=randomized_svd(Xc,n_components=128,random_state=0); Er=U*S
kf=KFold(5,shuffle=True,random_state=0)
def r2(X,y): return cross_val_score(make_pipeline(StandardScaler(),Ridge(alpha=10.)),X,y,cv=kf,scoring='r2',n_jobs=1).mean()
print(f"\n  predicting pixel-derived properties from the CACHED AION IMAGE EMBEDDING")
print(f"  {'property':<18}{'R^2':>9}   verdict")
for col in df.columns:
    y=df[col].values
    if not np.isfinite(y).all() or y.std()<1e-9: print(f'  {col:<18}{"n/a":>9}'); continue
    v=r2(Er,y)
    print(f"  {col:<18}{v:>9.3f}   {'ok' if v>0.3 else ('weak' if v>0.05 else 'NO SIGNAL')}")
