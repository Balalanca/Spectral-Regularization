import numpy as np
from original_main import *
from utils_fft import make_centered_frequency_grid

def test_identity():
    f=np.random.default_rng(1).normal(size=(16,16));_,_,w=make_centered_frequency_grid(16,16);k=np.ones_like(w);assert np.allclose(apply_blur_fft(f,k),f);assert np.allclose(tikhonov_reconstruct(f,k,w,0,'L2'),f)

def test_risk():
    f=np.random.default_rng(2).normal(size=(16,16));_,_,w=make_centered_frequency_grid(16,16);k=build_gaussian_khat_centered(w,1);g=apply_blur_fft(f,k);mu=.1;sig=.1;b,v,t=bias_variance_H1_analytic(f,k,w,np.array([mu]),sig);rng=np.random.default_rng(3);emp=np.mean([rel_l2_err(tikhonov_reconstruct(g+rng.normal(0,sig,g.shape),k,w,mu,'H1'),f)**2 for _ in range(150)]);assert np.isclose(emp,t[0],rtol=.1)
