"""Control: can AION's spectrum embedding predict REDSHIFT? Every spectrum representation
encodes z strongly. If AION can't, the way we call it is broken, not the readout."""
import os, sys, json, numpy as np, pandas as pd, torch, pyarrow.parquet as pq
sys.path.insert(0,'/work/hdd/bfvh/li26/Astrobridge-Train-v6/src')
from captioner.data.spectra_dataset import _canonical_object_id
from captioner.encoders.aion_spectrum import AionSpectrumEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
SNAP='/work/hdd/bfvh/li26/hf-cache/datasets--UniverseTBD--AstroBridge-Data/snapshots/3fe0bef6f7fa48894f0d2d7329df6559cd7ad87a'
D='/work/hdd/bfvh/li26/Astrobridge-Train-v6/diag'
lab=json.load(open(f'{D}/spec_labels.json'))
df=pq.read_table(f'{SNAP}/observations/spectra/desi_sdss_subset_crossmatch_nolan_1.0arcsec.parquet').to_pandas(ignore_metadata=True)
df['oid']=df['object_id'].map(_canonical_object_id); df=df.drop_duplicates('oid')
df['cls']=df['oid'].map(lambda o: lab.get(o,'?'))
df=df[(df['cls'].isin(['quasar','earlytype','starforming','agn']))&(df['survey']=='sdss')]
df=df[np.isfinite(df['Z'])&(df['Z']>-0.1)&(df['Z']<7)].head(700)
y=np.asarray(df['cls'].astype(str)); z=np.asarray(df['Z'],dtype=np.float64)
print(f'{len(df)} SDSS objects; z range {z.min():.3f}-{z.max():.3f}',flush=True)

BANDS=[(3700,3900),(4000,4200),(4400,4700),(5000,5300),(5600,5900),(6100,6400),(6500,6700),(7000,7400),(8000,8500)]
RAW=[]
for _,r in df.iterrows():
    s=r['spectrum']; fl=np.asarray(s['flux'],np.float32); wl=np.asarray(s['lambda'],np.float32); iv=np.asarray(s['ivar'],np.float32)
    g=(iv>0)&np.isfinite(fl); v=[]
    for a,b in BANDS:
        m=g&(wl>=a)&(wl<b); v.append(np.median(fl[m]) if m.sum()>5 else 0.0)
    v=np.array(v); v=v/(np.median(np.abs(v))+1e-9); RAW.append(v)
RAW=np.array(RAW)

enc=AionSpectrumEncoder('spectra','polymathic-ai/aion-base',None,{},512); enc.load('cuda')
def embed(nt):
    enc.num_encoder_tokens=nt; rows=[r for _,r in df.iterrows()]; E=[]
    for s in range(0,len(rows),32):
        ch=rows[s:s+32]; L=max(len(r['spectrum']['flux']) for r in ch); B=len(ch)
        fl=torch.zeros(B,L); wv=torch.zeros(B,L); iv=torch.zeros(B,L); mk=torch.ones(B,L,dtype=torch.bool)
        for i,r in enumerate(ch):
            sp=r['spectrum']; n=len(sp['flux'])
            fl[i,:n]=torch.from_numpy(np.asarray(sp['flux'],np.float32).copy())
            wv[i,:n]=torch.from_numpy(np.asarray(sp['lambda'],np.float32).copy())
            iv[i,:n]=torch.from_numpy(np.asarray(sp['ivar'],np.float32).copy())
            mk[i,:n]=torch.from_numpy(np.asarray(sp['mask'],bool).copy())
        with torch.no_grad(): e=enc.encode({'flux':fl,'wavelength':wv,'ivar':iv,'mask':mk,'survey':['sdss']*B})
        E.append(e.float().cpu().numpy())
    return np.concatenate(E)
E512=embed(512)
print('emb shape',E512.shape,flush=True)
Em=E512.mean(1); Ef=E512.reshape(len(E512),-1)

skf=StratifiedKFold(5,shuffle=True,random_state=0); kf=KFold(5,shuffle=True,random_state=0)
def cls_acc(X): return cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000)),X,y,cv=skf,n_jobs=1).mean()
def cls_mlp(X): return cross_val_score(make_pipeline(StandardScaler(),MLPClassifier((256,),max_iter=600,random_state=0)),X,y,cv=skf,n_jobs=1).mean()
def z_r2(X):  return cross_val_score(make_pipeline(StandardScaler(),Ridge(alpha=10.)),X,z,cv=kf,scoring='r2',n_jobs=1).mean()
maj=pd.Series(y).value_counts(normalize=True).max()
print(f"\n=== CLASS (majority={maj:.3f}) ===")
print(f"  raw flux, 9 medians        linear={cls_acc(RAW):.3f}   mlp={cls_mlp(RAW):.3f}")
print(f"  AION mean-pooled (768)     linear={cls_acc(Em):.3f}   mlp={cls_mlp(Em):.3f}")
print(f"  AION full grid (273x768)   linear={cls_acc(Ef):.3f}")
print(f"\n=== REDSHIFT z, Ridge R^2 (control: any real spectrum encoder should be high) ===")
print(f"  raw flux, 9 medians        R2={z_r2(RAW):.3f}")
print(f"  AION mean-pooled           R2={z_r2(Em):.3f}")
print(f"  AION full grid             R2={z_r2(Ef):.3f}")
