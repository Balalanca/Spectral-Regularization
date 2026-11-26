import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

from utils_fft import make_centered_frequency_grid



# fit the image to 256x256 pixels
def center_crop_or_pad(img: np.ndarray, n: int = 256, m: int = 256) -> np.ndarray:

    h, w = img.shape
    out = np.zeros((m, n), dtype=img.dtype)

    hs = min(h, m)
    ws = min(w, n)

    y0 = (h - hs) // 2
    x0 = (w - ws) // 2
    out_y0 = (m - hs) // 2
    out_x0 = (n - ws) // 2

    out[out_y0:out_y0 + hs, out_x0:out_x0 + ws] = img[y0:y0 + hs, x0:x0 + ws]
    return out


# here we load the image and ensure it is grayscale + normalize it
def load_grayscale_image(path: str, N: int) -> np.ndarray:
    
    img = iio.imread(path)

    # from RGB/RGBA to grayscale
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114

    img = img.astype(np.float64)

    # and normalize to [0,1]
    img -= img.min()
    if img.max() > 0:
        img /= img.max()

    return center_crop_or_pad(img, N, N)


# in order to achieve our target SNR in dB, we have to add some gaussian noise
# SNR_dB = 20 log10(sigma_signal / sigma_noise)
#    hence sigma_noise = 10^(-SNR_dB/20) * sigma_signal
def add_gaussian_noise(img: np.ndarray, snr_db: float, seed: int | None = None):
   
    rng = np.random.default_rng(seed)
    sigma_signal = np.std(img)
    sigma_noise = (10 ** (-snr_db / 20.0)) * sigma_signal
    noise = rng.normal(0.0, sigma_noise, size=img.shape)
    return img + noise, noise, sigma_noise



# function defined in order to help with saving the plots
# we save the current fugire if save_figs = true
def maybe_show(
    show: bool,
    save_figs: bool,
    fig_name: str | None = None,
    out_dir: str = "figures",
):
   
    
    if save_figs:
        os.makedirs(out_dir, exist_ok=True)
        if fig_name is None:    # If fig_name is None, autogenerate a unique filename.
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fig_name = f"figure_{ts}"
        path = os.path.join(out_dir, f"{fig_name}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[saved] {path}")

    if show:
        plt.show()
    else:
        plt.close()


# easy plotting function
def plot_four(
    images,
    titles,
    suptitle=None,
    show: bool = True,
    save_figs: bool = False,
    fig_name: str | None = None,
    out_dir: str = "figures",
):
    plt.figure(figsize=(14, 4))
    for i, (im, t) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, 4, i)
        plt.imshow(im, cmap="gray", vmin=0, vmax=1)
        plt.title(t)
        plt.axis("off")
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()
    maybe_show(show, save_figs, fig_name=fig_name, out_dir=out_dir)



# HEre we start the forward model and blur in Fourier domain ----

# Centered Gaussian transfer: Khat_c(w) = exp(-|w|^2 / (2*sigma^2)).
def build_gaussian_khat_centered(wmag_c: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(wmag_c ** 2) / (2 * sigma ** 2))


#  g_clean = ifft2( fft2(f_true) * Khat ), where Khat=ifftshift(Khat_c).
def apply_blur_fft(f_true: np.ndarray, Khat_c: np.ndarray) -> np.ndarray:
    Khat = np.fft.ifftshift(Khat_c)
    Fhat = np.fft.fft2(f_true)
    Ghat = Fhat * Khat
    return np.real(np.fft.ifft2(Ghat))


#Here we start the inverse model adn generalized Tikhonov
# Generalized Tikhonov in frequency domain:
#      fhat_mu = conj(Khat) / (|Khat|^2 + mu*|P|^2) * ghat
#    Penalties:
#      L2: |P|^2 = 1
#      H1: |P|^2 = |w|^2
#      H2: |P|^2 = |w|^4
def tikhonov_reconstruct(
    g: np.ndarray,
    Khat_c: np.ndarray,
    wmag_c: np.ndarray,
    mu: float,
    penalty: str
) -> np.ndarray:
    
    if penalty == "L2":
        Psq_c = np.ones_like(wmag_c)
    elif penalty == "H1":
        Psq_c = wmag_c ** 2
    elif penalty == "H2":
        Psq_c = (wmag_c ** 2) ** 2
    else:
        raise ValueError("Penalty must be 'L2', 'H1' or 'H2'")

    Khat = np.fft.ifftshift(Khat_c)
    Psq = np.fft.ifftshift(Psq_c)

    Ghat = np.fft.fft2(g)

    # so we can avoid division by very small numbers
    denom = np.abs(Khat) ** 2 + mu * Psq
    denom = np.maximum(denom, 1e-12)

    Fhat_mu = np.conj(Khat) * Ghat / denom
    return np.real(np.fft.ifft2(Fhat_mu))


#computes relative error ||a-b||2 / ||b||2
def rel_l2_err(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# Doing mu sweeps for each L2/H1/H2
# for each penalty: sweep mu over mu_grid, compute relative error,
# find best mu and reconstruction and produce 2 plots: erorr vs mu and best reconstruction

def run_sweep(
    g_obs: np.ndarray,
    f_true: np.ndarray,
    Khat_c: np.ndarray,
    wmag_c: np.ndarray,
    mu_grid: np.ndarray,
    snr_label: int,
    show_plots: bool = True,
    save_figs: bool = False,
    out_dir: str = "figures",
    penalties=None
):
    if penalties is None:
        penalties = ["L2", "H1", "H2"]

    errs = {pen: [] for pen in penalties}
    best = {}

    for pen in penalties:
        best_mu, best_e, best_rec = None, np.inf, None
        for mu in mu_grid:
            rec = tikhonov_reconstruct(g_obs, Khat_c, wmag_c, mu, pen)
            e = rel_l2_err(rec, f_true)
            errs[pen].append(e)
            if e < best_e:
                best_mu, best_e, best_rec = mu, e, rec
        best[pen] = (best_mu, best_e, best_rec)
        print(f"[{snr_label} dB] {pen}: best_mu={best_mu:.3e}, best_rel_err={best_e:.6f}")

    # plotting error vs mu
    plt.figure(figsize=(6, 4))
    for pen in penalties:
        plt.semilogx(mu_grid, errs[pen], label=pen)
    plt.xlabel("mu")
    plt.ylabel("relative L2 error")
    plt.title(f"Error vs mu (SNR = {snr_label} dB)")
    plt.legend()
    plt.tight_layout()
    maybe_show(
        show_plots,
        save_figs,
        fig_name=f"tikhonov_err_vs_mu_{snr_label}dB",
        out_dir=out_dir
    )

    # and showing best panels
    plot_four(
        [f_true, best["L2"][2], best["H1"][2], best["H2"][2]],
        [
            "true",
            f"L2 (mu={best['L2'][0]:.1e})",
            f"H1 (mu={best['H1'][0]:.1e})",
            f"H2 (mu={best['H2'][0]:.1e})",
        ],
        suptitle=f"Best reconstructions (SNR={snr_label} dB)",
        show=show_plots,
        save_figs=save_figs,
        fig_name=f"tikhonov_best_{snr_label}dB",
        out_dir=out_dir
    )

    return best, errs


# Part 3 spectral windowing 

#Hard spectral cutoff inversion:
#      Fhat = W * (Ghat / Khat)
#    with W=1 for |w|<=Omega, else 0.

def spectral_window_reconstruct(
    g: np.ndarray,
    Khat_c: np.ndarray,
    wmag_c: np.ndarray,
    Omega: float,
    eps: float = 1e-12
) -> np.ndarray:
   
    W_c = (wmag_c <= Omega).astype(np.float64)

    W = np.fft.ifftshift(W_c)
    Khat = np.fft.ifftshift(Khat_c)

    Ghat = np.fft.fft2(g)

    # for stability
    denom = np.where(np.abs(Khat) > eps, Khat, eps)
    Fhat = W * (Ghat / denom)

    return np.real(np.fft.ifft2(Fhat))



# we can also visualise how tihs mask looks like
def plot_spectral_window_mask(
    wmag_c: np.ndarray,
    Omega: float,
    show: bool = True,
    save_figs: bool = False,
    out_dir: str = "figures"
):
    W_c = (wmag_c <= Omega).astype(float)
    plt.figure(figsize=(5, 4))
    plt.imshow(W_c, cmap="gray")
    plt.title(f"Spectral window W (Omega={Omega})")
    plt.axis("off")
    plt.tight_layout()
    maybe_show(
        show,
        save_figs,
        fig_name=f"window_mask_Omega_{Omega:g}",
        out_dir=out_dir
    )


# and the reconstruction -> show observed image and reconstruct for multimple omegas
def plot_spectral_recons(
    g_obs: np.ndarray,
    Khat_c: np.ndarray,
    wmag_c: np.ndarray,
    Omegas: list[float],
    snr_label: int,
    show: bool = True,
    save_figs: bool = False,
    out_dir: str = "figures"
):
    images = [g_obs]
    titles = ["observed g"]

    for Omega in Omegas:
        images.append(spectral_window_reconstruct(g_obs, Khat_c, wmag_c, Omega))
        titles.append(f"Omega={Omega:g}")

    plt.figure(figsize=(14, 4))
    for i, (im, t) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, 1 + len(Omegas), i)
        plt.imshow(im, cmap="gray", vmin=0, vmax=1)
        plt.title(t)
        plt.axis("off")

    plt.suptitle(f"Spectral window reconstructions (SNR={snr_label} dB)")
    plt.tight_layout()
    maybe_show(
        show,
        save_figs,
        fig_name=f"spectral_recon_{snr_label}dB",
        out_dir=out_dir
    )



# we can start analyising the bias-variance tradeoff (H1 analytic)
# Analytic Bias^2 and Variance for H1 using FFT-domain formulas.
# Returns arrays (bias2, var, total).


def bias_variance_H1_analytic(
    f_true: np.ndarray,
    Khat_c: np.ndarray,
    wmag_c: np.ndarray,
    mu_grid: np.ndarray,
    sigma_noise: float,
    eps: float = 1e-12
):
    Fhat = np.fft.fft2(f_true)

    Khat = np.fft.ifftshift(Khat_c)
    wmag = np.fft.ifftshift(wmag_c)

    Psq = wmag ** 2  # H1 penalty
    Kabs2 = np.abs(Khat) ** 2

    denom_true = np.sum(np.abs(Fhat) ** 2) + eps
    norm_f2 = np.linalg.norm(f_true) ** 2 + eps

    bias2 = []
    var = []

    # we are using parseval's identity (Energy in spatial domain = energy in frequency domain (up to a constant factor))

    for mu in mu_grid:
        W = Kabs2 / (Kabs2 + mu * Psq + eps)
        H = np.conj(Khat) / (Kabs2 + mu * Psq + eps)

        b2 = np.sum(np.abs((W - 1.0) * Fhat) ** 2) / denom_true 
        bias2.append(b2)

        v = (sigma_noise ** 2) * np.sum(np.abs(H) ** 2) / norm_f2 
        var.append(v)

    bias2 = np.array(bias2)
    var = np.array(var)
    total = bias2 + var
    return bias2, var, total


def plot_bias_variance(
    mu_grid: np.ndarray,
    bias2: np.ndarray,
    var: np.ndarray,
    total: np.ndarray,
    snr_label: int,
    show: bool = True,
    save_figs: bool = False,
    out_dir: str = "figures"
):
    best_mu = mu_grid[np.argmin(total)]
    print(f"[Bias-Var H1 analytic] SNR={snr_label} dB: best_mu={best_mu:.3e}")

    plt.figure(figsize=(6, 4))
    plt.semilogx(mu_grid, bias2, label="Bias^2")
    plt.semilogx(mu_grid, var, label="Variance")
    plt.semilogx(mu_grid, total, label="Total")
    plt.yscale("log")
    plt.xlabel("mu")
    plt.ylabel("normalized (log scale)")
    plt.title(f"Bias-Variance Tradeoff (H1) — SNR={snr_label} dB")
    plt.legend()
    plt.tight_layout()
    maybe_show(
        show,
        save_figs,
        fig_name=f"bias_variance_H1_{snr_label}dB",
        out_dir=out_dir
    )





#    Build everything once: grid, f_true, Khat_c, g_clean, and noisy observations.
#    Returns a dict with all objects needed for the future parts.

def build_problem_instance(
    N: int,
    image_path: str,
    blur_sigma: float,
    snr_list: list[int],
    seed: int = 0
):
    
    wx_c, wy_c, wmag_c = make_centered_frequency_grid(N, N)
    f_true = load_grayscale_image(image_path, N)

    Khat_c = build_gaussian_khat_centered(wmag_c, blur_sigma)
    g_clean = apply_blur_fft(f_true, Khat_c)

    noisy = {}
    for snr_db in snr_list:
        g_obs, _, sigma_noise = add_gaussian_noise(g_clean, snr_db, seed=seed)
        noisy[snr_db] = (g_obs, sigma_noise)

    return {
        "N": N,
        "wx_c": wx_c,
        "wy_c": wy_c,
        "wmag_c": wmag_c,
        "f_true": f_true,
        "Khat_c": Khat_c,
        "g_clean": g_clean,
        "noisy": noisy,  # snr_db -> (g_obs, sigma_noise), stores both observation and nosie level
    }



def main():
    # main configuration
    cfg = {
        "N": 256,
        "image_path": "GreeceGrayScaled.png",
        "blur_sigma": 1.0,          # radians
        "snr_levels": [40, 20],
        "seed": 0,
        "mu_grid": np.logspace(-8, 0, 25),

        # we can decide which parts to run
        "RUN_PART2_SWEEPS": True,
        "RUN_PART3_SPECTRAL": True,
        "RUN_PART4_BIASVAR": True,

        # Plot control
        "SHOW_PLOTS": True,
        "SAVE_FIGS": False,
        "FIG_DIR": "figures",

        # Part 3 settings for spectral
        "OMEGAS_TO_SHOW": [1.0, 2.0, 4.0],
        "OMEGA_MASK_DEMO": 2.0,
    }

    # Build data only once 
    data = build_problem_instance(
        N=cfg["N"],
        image_path=cfg["image_path"],
        blur_sigma=cfg["blur_sigma"],
        snr_list=cfg["snr_levels"],
        seed=cfg["seed"],
    )

    f_true = data["f_true"]
    Khat_c = data["Khat_c"]
    wmag_c = data["wmag_c"]

    # Print noise levels
    for snr_db in cfg["snr_levels"]:
        _, sigma_noise = data["noisy"][snr_db]
        print(f"SNR target {snr_db} dB -> sigma_noise = {sigma_noise:.6f}")

    # Part 3 spectral windowing 
    if cfg["RUN_PART3_SPECTRAL"]:
        plot_spectral_window_mask(
            wmag_c=wmag_c,
            Omega=cfg["OMEGA_MASK_DEMO"],
            show=cfg["SHOW_PLOTS"],
            save_figs=cfg["SAVE_FIGS"],
            out_dir=cfg["FIG_DIR"],
        )

        for snr_db in cfg["snr_levels"]:
            g_obs, _ = data["noisy"][snr_db]
            plot_spectral_recons(
                g_obs=g_obs,
                Khat_c=Khat_c,
                wmag_c=wmag_c,
                Omegas=cfg["OMEGAS_TO_SHOW"],
                snr_label=snr_db,
                show=cfg["SHOW_PLOTS"],
                save_figs=cfg["SAVE_FIGS"],
                out_dir=cfg["FIG_DIR"],
            )

    # Part 2 sweeps
    if cfg["RUN_PART2_SWEEPS"]:
        for snr_db in cfg["snr_levels"]:
            g_obs, _ = data["noisy"][snr_db]
            run_sweep(
                g_obs=g_obs,
                f_true=f_true,
                Khat_c=Khat_c,
                wmag_c=wmag_c,
                mu_grid=cfg["mu_grid"],
                snr_label=snr_db,
                show_plots=cfg["SHOW_PLOTS"],
                save_figs=cfg["SAVE_FIGS"],
                out_dir=cfg["FIG_DIR"],
            )

    # Part 4 bias–variance (H1 analytic)
    if cfg["RUN_PART4_BIASVAR"]:
        for snr_db in cfg["snr_levels"]:
            _, sigma_noise = data["noisy"][snr_db]
            bias2, var, total = bias_variance_H1_analytic(
                f_true=f_true,
                Khat_c=Khat_c,
                wmag_c=wmag_c,
                mu_grid=cfg["mu_grid"],
                sigma_noise=sigma_noise
            )
            plot_bias_variance(
                mu_grid=cfg["mu_grid"],
                bias2=bias2,
                var=var,
                total=total,
                snr_label=snr_db,
                show=cfg["SHOW_PLOTS"],
                save_figs=cfg["SAVE_FIGS"],
                out_dir=cfg["FIG_DIR"],
            )


if __name__ == "__main__":
    main()
