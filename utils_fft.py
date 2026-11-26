import numpy as np

def make_centered_frequency_grid(n,m):
    x=2*np.pi*np.fft.fftshift(np.fft.fftfreq(n));y=2*np.pi*np.fft.fftshift(np.fft.fftfreq(m));xx,yy=np.meshgrid(x,y);return xx,yy,np.hypot(xx,yy)
