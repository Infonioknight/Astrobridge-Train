"""Encode the SAME SDSS spectra two ways and probe both:
  (A) BUGGY  - padded to 7781 with wavelength=0, ivar=1, mask=False  (what the cache did)
  (B) FIXED  - homogeneous batch, no cross-survey padding
  (C) PAD-OK - padded to 7781 but with mask=True / ivar=0 in the pad region
"""
import os, sys, json, numpy as np, pandas as pd, torch, pyarrow.parquet as pq
sys.path.insert(0,'/work/hdd/bfvh/li26/Astrobridge-Train-v6/src')
from captioner.data.spectra_dataset import _canonical_object_id
from captioner.encoders.aion_spectrum import AionSpectrumEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
SNAP='/work/hdd/bfvh/li26/hf-cache/datasets--UniverseTBD--AstroBridge-Data/snapshots/3fe0bef6f7fa48894f0d2d7329df6559cd7ad87a'
LAB='/work/hdd/bfvh/li26/Astrobridge-Train-v6/diag/spec_labels.json'
N=int(os.environ.get('NOBJ','600')); DEV='cuda'
lab=json.load(open(LAB))
df=pq.read_table(f'{SNAP}/observations/spectra/desi_sdss_subset_crossmatch_nolan_1.0arcsec.parquet').to_pandas(ignore_metadata=True)
df['oid']=df['object_id'].map(_canonical_object_id); df=df.drop_duplicates('oid')
df['cls']=df['oid'].map(lambda o: lab.get(o,'?'))
df=df[(df['cls'].isin(['quasar','earlytype','starforming','agn']))&(df['survey']=='sdss')].head(N)
print(f'{len(df)} SDSS objects; classes={dict(df["cls"].value_counts())}',flush=True)
enc=AionSpectrumEncoder('spectra','polymathic-ai/aion-base',None,{},512); enc.load(DEV)

def batch(rows, pad_to, pad_valid):
    B=len(rows); L=pad_to
    fl=torch.zeros(B,L); wv=torch.zeros(B,L)
    iv=torch.ones(B,L) if pad_valid else torch.zeros(B,L)
    mk=torch.zeros(B,L,dtype=torch.bool) if pad_valid else torch.ones(B,L,dtype=torch.bool)
    for i,r in enumerate(rows):
        s=r['spectrum']; n=len(s['flux'])
        fl[i,:n]=torch.from_numpy(np.asarray(s['flux'],np.float32))
        wv[i,:n]=torch.from_numpy(np.asarray(s['lambda'],np.float32))
        iv[i,:n]=torch.from_numpy(np.asarray(s['ivar'],np.float32))
        mk[i,:n]=torch.from_numpy(np.asarray(s['mask'],bool))
    return {'flux':fl,'wavelength':wv,'ivar':iv,'mask':mk,'survey':['sdss']*B}

def run(pad_to_fn, pad_valid, tag):
    E=[]
    rows=[r for _,r in df.iterrows()]
    for s in range(0,len(rows),32):
        ch=rows[s:s+32]
        L=pad_to_fn(ch)
        with torch.no_grad(): e=enc.encode(batch(ch,L,pad_valid))
        E.append(e.float().mean(1).cpu().numpy())
    E=np.concatenate(E); print(f'  {tag}: emb{E.shape}',flush=True); return E

y=np.asarray(df['cls'].astype(str)); cv=StratifiedKFold(5,shuffle=True,random_state=0)
maj=pd.Series(y).value_counts(normalize=True).max()
def probe(X): return cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000)),X,y,cv=cv,n_jobs=1).mean()

print('\n=== encoding ===',flush=True)
A=run(lambda ch: 7781, True,  'A BUGGY  pad->7781, wavelength=0 ivar=1 mask=False')
B=run(lambda ch: max(len(r['spectrum']['flux']) for r in ch), True, 'B FIXED  homogeneous SDSS batch')
C=run(lambda ch: 7781, False, 'C PAD-OK pad->7781 but mask=True ivar=0')
print(f'\n=== linear probe, {len(y)} objects, majority={maj:.3f} ===')
for tag,X in [('A BUGGY  (what v5/v6 trained on)',A),('B FIXED  (homogeneous batch)',B),('C PAD-OK (padded, masked properly)',C)]:
    print(f'  {tag:<38} acc={probe(X):.3f}')
d=np.linalg.norm(A-B,axis=1)/ (np.linalg.norm(B,axis=1)+1e-9)
print(f'\n  mean relative ||A-B||/||B|| = {d.mean():.3f}  (0 = padding had no effect)')
