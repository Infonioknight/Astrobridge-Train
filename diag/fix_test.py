"""CONFIRMATION: trim the -1 wavelength sentinels (keep a strictly-increasing grid) and
re-encode. If token diversity and probe accuracy jump, that was the whole bug."""
import json,sys,numpy as np,pandas as pd,torch,pyarrow.parquet as pq
sys.path.insert(0,'/work/hdd/bfvh/li26/Astrobridge-Train-v6/src')
from captioner.data.spectra_dataset import _canonical_object_id
from captioner.encoders.aion_spectrum import AionSpectrumEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
SNAP='/work/hdd/bfvh/li26/hf-cache/datasets--UniverseTBD--AstroBridge-Data/snapshots/3fe0bef6f7fa48894f0d2d7329df6559cd7ad87a'
D='/work/hdd/bfvh/li26/Astrobridge-Train-v6/diag'
lab=json.load(open(f'{D}/spec_labels.json'))
df=pq.read_table(f'{SNAP}/observations/spectra/desi_sdss_subset_crossmatch_nolan_1.0arcsec.parquet').to_pandas(ignore_metadata=True)
df['oid']=df['object_id'].map(_canonical_object_id); df=df.drop_duplicates('oid')
df['cls']=df['oid'].map(lambda o: lab.get(o,'?'))
df=df[(df['cls'].isin(['quasar','earlytype','starforming','agn']))&(df['survey']=='sdss')].head(700)
y=np.asarray(df['cls'].astype(str)); rows=[r for _,r in df.iterrows()]
print(f'{len(rows)} SDSS objects {dict(pd.Series(y).value_counts())}',flush=True)
enc=AionSpectrumEncoder('spectra','polymathic-ai/aion-base',None,{},512); enc.load('cuda')
from aion.modalities import SDSSSpectrum

def arrays(r, trim):
    s=r['spectrum']
    fl=np.asarray(s['flux'],np.float32).copy(); wl=np.asarray(s['lambda'],np.float32).copy()
    iv=np.asarray(s['ivar'],np.float32).copy(); mk=np.asarray(s['mask'],bool).copy()
    if trim:
        g=wl>0                                   # drop the -1 sentinels
        fl,wl,iv,mk=fl[g],wl[g],iv[g],mk[g]
        o=np.argsort(wl); fl,wl,iv,mk=fl[o],wl[o],iv[o],mk[o]   # guarantee strictly increasing
    return fl,wl,iv,mk

def encode(trim):
    E=[];T=[]
    for s in range(0,len(rows),24):
        ch=rows[s:s+24]; A=[arrays(r,trim) for r in ch]; L=max(len(a[0]) for a in A); B=len(ch)
        fl=torch.zeros(B,L); wv=torch.zeros(B,L); iv=torch.zeros(B,L); mk=torch.ones(B,L,dtype=torch.bool)
        for i,(f,w,v,m) in enumerate(A):
            n=len(f); fl[i,:n]=torch.from_numpy(f); iv[i,:n]=torch.from_numpy(v); mk[i,:n]=torch.from_numpy(m)
            wv[i,:n]=torch.from_numpy(w)
            if n<L and trim:  # keep the padded tail STRICTLY INCREASING so searchsorted stays valid
                wv[i,n:]=torch.from_numpy(w[-1]+0.8*np.arange(1,L-n+1,dtype=np.float32))
        b={'flux':fl,'wavelength':wv,'ivar':iv,'mask':mk,'survey':['sdss']*B}
        with torch.no_grad():
            e=enc.encode(b); E.append(e.float().mean(1).cpu().numpy())
            from captioner.encoders.aion_common import load_aion
            mdl,cm=load_aion('polymathic-ai/aion-base',None,'cuda')
            t=cm.encode(SDSSSpectrum(flux=fl.cuda(),ivar=iv.cuda(),mask=mk.cuda(),wavelength=wv.cuda()))
            T.append(list(t.values())[0].cpu().numpy())
    return np.concatenate(E), np.concatenate(T)

cv=StratifiedKFold(5,shuffle=True,random_state=0)
maj=pd.Series(y).value_counts(normalize=True).max()
def lin(X): return cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000)),X,y,cv=cv,n_jobs=1).mean()
def mlp(X): return cross_val_score(make_pipeline(StandardScaler(),MLPClassifier((256,),max_iter=600,random_state=0)),X,y,cv=cv,n_jobs=1).mean()
for tag,trim in [('CURRENT  (-1 sentinels left in)',False),('TRIMMED  (strictly increasing wl)',True)]:
    E,T=encode(trim)
    ident=np.mean([(T[i]==T[j]).mean() for i in range(0,len(T),23) for j in range(0,len(T),29) if i!=j])
    perpos=np.mean([len(np.unique(T[:,j])) for j in range(T.shape[1])])
    print(f'\n{tag}')
    print(f'   identical-token fraction between objects: {ident:.3f}   distinct codes/position: {perpos:.1f}')
    print(f'   class probe: linear={lin(E):.3f}  mlp={mlp(E):.3f}   (majority={maj:.3f})',flush=True)
