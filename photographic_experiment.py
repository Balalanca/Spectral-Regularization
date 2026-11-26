"""Photographic deblurring benchmark and spectral-cutoff sensitivity."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
from skimage import data,transform
from original_main import *
from utils_fft import make_centered_frequency_grid

def main():
    p=argparse.ArgumentParser();p.add_argument('--image');a=p.parse_args();out='results/photographic';Path(out).mkdir(parents=True,exist_ok=True)
    f=load_grayscale_image(a.image,256) if a.image else transform.resize(data.camera(),(256,256),anti_aliasing=True)
    _,_,w=make_centered_frequency_grid(256,256);k=build_gaussian_khat_centered(w,1.0);g=apply_blur_fft(f,k);mus=np.logspace(-8,0,25)
    plot_spectral_window_mask(w,2.,False,True,out)
    for snr in (20,40):
        obs,_,sigma=add_gaussian_noise(g,snr,42)
        run_sweep(obs,f,k,w,mus,snr,False,True,out)
        plot_spectral_recons(obs,k,w,[1.,2.,4.],snr,False,True,out)
        b,v,t=bias_variance_H1_analytic(f,k,w,mus,sigma)
        plot_bias_variance(mus,b,v,t,snr,False,True,out)
if __name__=='__main__':main()
