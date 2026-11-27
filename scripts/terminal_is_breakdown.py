# Descriptive: split of the analytic isotropic log(chi^2+1) per basin into variance and mean-error terms (terminal_is.py, same halves). Run from the repo root.
import json,math,numpy as np,sys
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import gmm_dk
R='results/darting_large_sigma'; D=128
mu=gmm_dk.means(D,8,6.0).double().numpy(); out={}
for c in ('d128e1000','d128e1250','d128'):
    x=np.concatenate([np.load(f'{R}/sim_{c}_s{s}_p0of1.npz')['x'] for s in (1,2,3)]).astype(float); lab=np.concatenate([np.load(f'{R}/raw_{c}_s{s}.npz')['basins'] for s in (1,2,3)])
    fit=np.random.default_rng(4242).permutation(len(x))[:len(x)//2]; rows=[]
    for k in range(8):
        xf=x[fit][lab[fit]==k]; m=xf.mean(0); r=float(xf.var(0,ddof=1).mean()); d2=float(((m-mu[k])**2).sum()); noise=D*r/len(xf); v=D*math.log(r/math.sqrt(2*r-1)); mm=d2/(2*r-1)
        rows.append(dict(k=k,n=len(xf),r=r,mean_err_sq=d2,mean_noise_floor=noise,log_term_var=v,log_term_mean=mm,log_term_mean_debiased=max(d2-noise,0)/(2*r-1)))
    out[c]=rows
    A={q:float(np.mean([z[q] for z in rows])) for q in ('r','mean_err_sq','mean_noise_floor','log_term_var','log_term_mean','log_term_mean_debiased')}; print(c,{k:round(v,3) for k,v in A.items()})
json.dump(out,open('figures/imh_path/terminal_iso_breakdown.json','w'),indent=1)
print('user estimate (r/sqrt(2r-1))^d at r=1.08:',(1.08/math.sqrt(1.16))**128,' r=1.1:',(1.1/math.sqrt(1.2))**128)
