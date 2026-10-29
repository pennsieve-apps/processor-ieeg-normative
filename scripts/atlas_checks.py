"""
Checks that rule out a bug behind the near-chance AUCs of scripts/validate_atlas.py:

1. the atlas arrays index the same channels (signal correlation of channel pairs falls with
   MNI distance; spike_24h tracks the spectrum; SOZ labels cluster in space);
2. replication grid: RADIUS_MM 20/30/40 x leave-one-patient-out on/off x irritative zone kept/
   excluded, max vs mean |z| over bands, SOZ vs all other channels and vs normative channels,
   resected SOZ (the paper's classifier target);
3. per band |z| of SOZ vs normative channels (two-sided rank-sum), and the median signed z.

  PYTHONPATH=. python scripts/atlas_checks.py   (path to HUP_atlas_final.mat in ATLAS, default below)
"""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np, scipy.io as sio
from scipy import stats
from pathlib import Path
from processor import atlas as A, normative as N, spectral as S
P=os.environ.get('ATLAS', '../workflow-epilepsy-ieeg/data/atlas/HUP_atlas_final.mat')
for n,shape,cls in sio.whosmat(P): print('whosmat',n,shape,cls)
at=A.load(Path(P)); W=at.wake_clip
print('wake_clip',W.shape,'std per channel median %.2f range %.2f-%.2f'%(np.median(W.std(0)),W.std(0).min(),W.std(0).max()))
# 1a. consecutive channels within a patient: coordinate distance vs signal correlation
same=at.patient[1:]==at.patient[:-1]
d=np.linalg.norm(at.xyz[1:]-at.xyz[:-1],axis=1)[same]
Wc=(W-W.mean(0))/W.std(0)
r=(Wc[:,1:]*Wc[:,:-1]).mean(0)[same]
print('consecutive pairs: median dist %.1f mm; |corr| median %.2f'%(np.median(d),np.median(np.abs(r))))
print('  corr(|signal corr|, distance) = %.3f (expect negative)'%stats.spearmanr(np.abs(r),d)[0])
# shuffle control: random pairs within patient
rng=np.random.default_rng(0); idx=np.arange(at.n_channels); pr=[]
for p in np.unique(at.patient):
    m=idx[at.patient==p]; a=rng.choice(m,200); b=rng.choice(m,200); k=a!=b
    pr.append((np.linalg.norm(at.xyz[a[k]]-at.xyz[b[k]],axis=1),np.abs((Wc[:,a[k]]*Wc[:,b[k]]).mean(0))))
dd=np.concatenate([x[0] for x in pr]); rr=np.concatenate([x[1] for x in pr])
print('  random within-patient pairs: spearman(|corr|,dist) = %.3f'%stats.spearmanr(rr,dd)[0])
# misaligned control: shift wake_clip columns by 1 relative to coords
# 1b. spike rate vs power
t=A.reference_features(at,S.SpectralParams())
iz=at.spike_24h>24
for b,name in enumerate(t.bands):
    print('  %-9s spearman(log spikes, log abs)=%.3f  rel=%.3f ; IZ vs non-IZ rel median %.3f/%.3f p=%.1e'%(name,
      stats.spearmanr(np.log1p(at.spike_24h),np.log10(t.abs_power[:,b]))[0], stats.spearmanr(np.log1p(at.spike_24h),t.rel_power[:,b])[0],
      np.median(t.rel_power[iz,b]),np.median(t.rel_power[~iz,b]), stats.mannwhitneyu(t.rel_power[iz,b],t.rel_power[~iz,b]).pvalue))
# soz spatial clustering within patient: soz channels nearer to other soz than to non-soz
ds=[];dn=[]
for p in np.unique(at.patient):
    m=at.patient==p; s=at.soz[m]; X=at.xyz[m]
    if s.sum()<2 or (~s).sum()<1: continue
    D=np.linalg.norm(X[:,None]-X[None],axis=2)
    for i in np.where(s)[0]:
        o=np.where(s)[0]; o=o[o!=i]; ds.append(D[i,o].min()); dn.append(D[i,~s].min())
print('SOZ channel: nearest other SOZ %.1f mm vs nearest non-SOZ %.1f mm (median)'%(np.median(ds),np.median(dn)))
print('soz subset of resected? soz&~res =',int((at.soz&~at.resected).sum()),' soz&res =',int((at.soz&at.resected).sum()))
iz=at.spike_24h>24
t=A.reference_features(at,S.SpectralParams())
v=N.transform('norm_median',t.abs_power[:,:5],t.rel_power[:,:5],t.norm_median[:,:5])
bands=t.bands[:5]
def Zall(R,lopo,exiz):
    norm=~at.soz&~at.resected
    if exiz: norm&=~iz
    ref=N.Reference(at.xyz[norm],at.patient[norm],v[norm])
    Z=np.full((at.n_channels,5),np.nan)
    for p in np.unique(at.patient):
        m=at.patient==p
        Z[m]=N.zscore(ref,at.xyz[m],v[m],p if lopo else None,R,10).z
    return Z,norm
print('R lopo exIZ | AUC SOZ vs rest: max|z| mean|z| | AUC SOZ vs normative: max mean | AUC resected-SOZ (Bernabei target) vs rest: max mean | scored')
for R in (20,30,40):
  for lopo in (True,False):
    for exiz in (False,True):
      Z,norm=Zall(R,lopo,exiz); A_=np.abs(Z); mx=np.nanmax(A_,1); mn=np.nanmean(A_,1)
      sel=at.soz|norm; tgt=at.soz&at.resected
      print(R,lopo,exiz,'| %.3f %.3f | %.3f %.3f | %.3f %.3f | %d'%(N.auc(mx,at.soz),N.auc(mn,at.soz),N.auc(mx[sel],at.soz[sel]),N.auc(mn[sel],at.soz[sel]),N.auc(mx,tgt),N.auc(mn,tgt),np.isfinite(mx).sum()))
# per band |z| SOZ vs normative channels, rank-sum (R=20, LOPO, exclude IZ as in paper)
for exiz in (False,True):
  Z,norm=Zall(20,True,exiz)
  print('per band |z| (R=20, LOPO, exclude_IZ=%s): SOZ vs normative channels, median |z| and two-sided ranksum p; IZ vs normative'%exiz)
  for b in range(5):
    a=np.abs(Z[at.soz,b]); c=np.abs(Z[norm,b]); i=np.abs(Z[iz&~at.soz&~at.resected,b])
    a,c,i=a[np.isfinite(a)],c[np.isfinite(c)],i[np.isfinite(i)]
    print('  %-6s SOZ %.2f vs norm %.2f p=%.2g | IZ-only %.2f p=%.2g | signed z SOZ median %+.2f'%(bands[b],np.median(a),np.median(c),stats.mannwhitneyu(a,c).pvalue,np.median(i),stats.mannwhitneyu(i,c).pvalue,np.nanmedian(Z[at.soz,b])))
