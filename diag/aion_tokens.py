"""Is AION's spectrum TOKENIZER working on our inputs? Two self-contained checks:
   1. token diversity  - do a quasar and an early-type galaxy get different codes?
   2. round-trip       - encode->decode; does the reconstruction match the input flux?
If both fail, we are feeding AION wrong. If both pass, AION's codes are fine and the loss is later."""
import json,sys,numpy as np,pandas as pd,torch,pyarrow.parquet as pq
sys.path.insert(0,'/work/hdd/bfvh/li26/Astrobridge-Train-v6/src')
from captioner.data.spectra_dataset import _canonical_object_id
from captioner.encoders.aion_common import load_aion
SNAP='/work/hdd/bfvh/li26/hf-cache/datasets--UniverseTBD--AstroBridge-Data/snapshots/3fe0bef6f7fa48894f0d2d7329df6559cd7ad87a'
D='/work/hdd/bfvh/li26/Astrobridge-Train-v6/diag'
lab=json.load(open(f'{D}/spec_labels.json'))
df=pq.read_table(f'{SNAP}/observations/spectra/desi_sdss_subset_crossmatch_nolan_1.0arcsec.parquet').to_pandas(ignore_metadata=True)
df['oid']=df['object_id'].map(_canonical_object_id); df=df.drop_duplicates('oid')
df['cls']=df['oid'].map(lambda o: lab.get(o,'?'))
df=df[(df['cls'].isin(['quasar','earlytype','starforming']))&(df['survey']=='sdss')]
sel=pd.concat([df[df['cls']==c].head(60) for c in ['quasar','earlytype','starforming']])
print(f'{len(sel)} objects: {dict(sel["cls"].value_counts())}',flush=True)
model,cm=load_aion('polymathic-ai/aion-base',None,'cuda')
from aion.modalities import SDSSSpectrum
rows=[r for _,r in sel.iterrows()]; TOK=[]; REC=[]
for s in range(0,len(rows),20):
    ch=rows[s:s+20]; L=max(len(r['spectrum']['flux']) for r in ch); B=len(ch)
    fl=torch.zeros(B,L); wv=torch.zeros(B,L); iv=torch.zeros(B,L); mk=torch.ones(B,L,dtype=torch.bool)
    for i,r in enumerate(ch):
        sp=r['spectrum']; n=len(sp['flux'])
        fl[i,:n]=torch.from_numpy(np.asarray(sp['flux'],np.float32).copy())
        wv[i,:n]=torch.from_numpy(np.asarray(sp['lambda'],np.float32).copy())
        iv[i,:n]=torch.from_numpy(np.asarray(sp['ivar'],np.float32).copy())
        mk[i,:n]=torch.from_numpy(np.asarray(sp['mask'],bool).copy())
    spec=SDSSSpectrum(flux=fl.cuda(),ivar=iv.cuda(),mask=mk.cuda(),wavelength=wv.cuda())
    with torch.no_grad():
        t=cm.encode(spec); key=[k for k in t][0]; TOK.append(t[key].cpu().numpy())
        codec=cm._codecs[key] if hasattr(cm,'_codecs') else None
        if codec is not None:
            try: REC.append(codec.decode(t[key],wavelength=wv.cuda()).flux.cpu().numpy())
            except Exception as e: print('decode failed:',e); REC=None; codec=None
T=np.concatenate(TOK); y=sel['cls'].values
print(f'\ntokens shape={T.shape} dtype={T.dtype} vocab_used={len(np.unique(T))} min={T.min()} max={T.max()}',flush=True)
print(f'per-position distinct codes across objects: mean={np.mean([len(np.unique(T[:,j])) for j in range(T.shape[1])]):.1f}')
same=np.mean([(T[i]==T[j]).mean() for i in range(0,len(T),7) for j in range(0,len(T),11) if i!=j])
print(f'mean fraction of IDENTICAL token positions between two objects: {same:.3f}   (1.0 = tokenizer collapsed)')
for c in ['quasar','earlytype','starforming']:
    idx=np.where(y==c)[0]; print(f'  {c:<12} within-class identical-token frac={np.mean([(T[i]==T[j]).mean() for i in idx[:12] for j in idx[:12] if i!=j]):.3f}')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
oh=OneHotEncoder(handle_unknown='ignore',max_categories=64).fit_transform(T)
print(f'\nlinear probe on RAW TOKEN CODES (one-hot): acc={cross_val_score(LogisticRegression(max_iter=3000),oh,y,cv=5,n_jobs=1).mean():.3f} (majority={pd.Series(y).value_counts(normalize=True).max():.3f})')
