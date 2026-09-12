"""Class probe on whatever is currently in outputs/cache/spectra — the production cache."""
import os, sys, json
for v in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[v]='8'
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src'))
from pathlib import Path
from captioner.data.dataset import ModalityCacheReader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import randomized_svd

root=Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/cache')
lab=json.load(open(Path(__file__).parent/'spec_labels.json'))
m=pd.read_parquet('outputs/manifest/manifest.parquet')
h=[p.name for p in (root/'spectra').iterdir() if p.is_dir()][0]
rd=ModalityCacheReader(root,'spectra',h)
ids=[o for o in m.loc[m['has_spectra'],'object_id'] if o in rd.index.index
     and str(lab.get(o)) in ('quasar','earlytype','starforming','agn')][:900]
y=np.array([lab[o] for o in ids]); E=np.stack([rd.get(o).mean(0) for o in ids])
Xc=E-E.mean(0,keepdims=True); U,S,_=randomized_svd(Xc,n_components=128,random_state=0); Er=U*S
cv=StratifiedKFold(5,shuffle=True,random_state=0)
maj=pd.Series(y).value_counts(normalize=True).max()
lin=cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000)),Er,y,cv=cv,n_jobs=1).mean()
mlp=cross_val_score(make_pipeline(StandardScaler(),MLPClassifier((256,),max_iter=600,random_state=0)),Er,y,cv=cv,n_jobs=1).mean()
print(f"{root}/spectra/{h}: n={len(y)} classes={sorted(set(y))}")
print(f"   linear={lin:.3f}  mlp={mlp:.3f}   (majority={maj:.3f})")
