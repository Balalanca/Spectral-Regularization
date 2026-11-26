from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from original_main import apply_blur_fft,build_gaussian_khat_centered,add_gaussian_noise,tikhonov_reconstruct,rel_l2_err,bias_variance_H1_analytic,spectral_window_reconstruct
from utils_fft import make_centered_frequency_grid

def phantom(n=256):
    y,x=np.mgrid[-1:1:complex(n),-1:1:complex(n)];return .2+.45*((x+.2)**2+(y+.1)**2<.35**2)+.3*((abs(x-.35)<.12)&(abs(y-.2)<.45))+.04*np.sin(18*x)*np.cos(12*y)

def main():
    out=Path('results');out.mkdir(exist_ok=True);f=phantom();_,_,w=make_centered_frequency_grid(256,256);k=build_gaussian_khat_centered(w,.7);g=apply_blur_fft(f,k);mus=np.logspace(-8,0,25)
    for snr in [20,40]:
        obs,_,sigma=add_gaussian_noise(g,snr,42);fig,ax=plt.subplots();best=[]
        for pen in ['L2','H1','H2']:
            errs=[rel_l2_err(tikhonov_reconstruct(obs,k,w,mu,pen),f) for mu in mus];i=np.argmin(errs);ax.loglog(mus,errs,label=pen);best.append(tikhonov_reconstruct(obs,k,w,mus[i],pen));print(snr,pen,mus[i],errs[i])
        ax.set(xlabel='Regularization strength',ylabel='Relative L2 error');ax.legend();fig.savefig(out/f'error_{snr}.png');plt.close(fig)
        fig,axes=plt.subplots(1,6,figsize=(13,3));images=[f,obs,*best,spectral_window_reconstruct(obs,k,w,1.5)];
        for a,im,title in zip(axes,images,['Truth','Observed','L2','H1','H2','Cutoff']):a.imshow(im,cmap='gray',vmin=0,vmax=1);a.set_title(title);a.axis('off')
        fig.tight_layout();fig.savefig(out/f'reconstruction_{snr}.png');plt.close(fig)
        b,v,total=bias_variance_H1_analytic(f,k,w,mus,sigma);i=np.argmin(total);emp=np.mean([rel_l2_err(tikhonov_reconstruct(g+np.random.default_rng(j).normal(0,sigma,g.shape),k,w,mus[i],'H1'),f)**2 for j in range(30)]);print('H1 risk theory/MC',total[i],emp)
        fig,ax=plt.subplots();ax.loglog(mus,b,label='Bias squared');ax.loglog(mus,v,label='Variance');ax.loglog(mus,total,label='Total');ax.scatter(mus[i],emp,label='Monte Carlo');ax.legend();ax.set(xlabel='Regularization strength',ylabel='Normalized squared error');fig.savefig(out/f'risk_{snr}.png');plt.close(fig)
if __name__=='__main__':main()
