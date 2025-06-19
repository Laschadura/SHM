# diff_pt/tests/test_reconstruction.py
"""
Compare original time-series segments with signals reconstructed
through the trained spectrogram auto-encoder (specAE).

Run:
    python -m diff_pt.tests.test_reconstruction --ckpt_dir results_diff --n_segments 1
"""

import argparse, os, numpy as np, pandas as pd, torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from bridge_data.loader      import load_data
from bridge_data.preprocess  import compute_complex_spectrogram
from bridge_data.postprocess import inverse_spectrogram, align_by_xcorr
from bridge_data.tests.utils import (
    calculate_reconstruction_metrics,
    plot_overlay_all_channels,
)

from diff_pt import model as dm


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir",   default="results_diff")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_segments", type=int, default=3)
    p.add_argument("--fs",     type=int, default=200)
    p.add_argument("--nperseg",type=int, default=256)
    p.add_argument("--noverlap",type=int, default=224)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# MAIN                                                                        #
# --------------------------------------------------------------------------- #
def main():
    args   = _parse()
    device = torch.device(args.device)
    ckpt   = Path(args.ckpt_dir)

    # --------------------------------------------------------------------- #
    # 1.  Load cached, *normalised* windows                                 #
    # --------------------------------------------------------------------- #
    _, _, _, segments, _, test_ids, metadata, *_ = load_data(recompute=False)

    # keep the very same snippets as the reference implementation
    segments  = segments[: args.n_segments]
    test_ids  = test_ids[: args.n_segments]
    metadata  = metadata[: args.n_segments]

    # --------------------------------------------------------------------- #
    # 2.  Bring the trained spec-AE back to life (weights + μ/σ)            #
    # --------------------------------------------------------------------- #
    mld = dm.MultiModalLatentDiffusion.from_checkpoint(
        ae_dir        = ckpt / "autoencoders",
        diff_ckpt_path= ckpt / "diffusion" / "final_diffusion_model.pt",
        spec_norm_path= "cache/spec_norm_magonly.npz",
        latent_dim    = 384,
        device        = device,
    )
    specAE = mld.autoencoders["spec"].eval()       # just the auto-encoder
    mu     = mld.mu_spec.to(device)                # (2C, F, T) broadcast-able
    sigma  = mld.sig_spec.to(device)

    print("🔍 Checking encoder weights:")
    print("‣ enc_mag.dw_freq ‖w‖ =",
      specAE.enc_mag.dw_freq.weight.norm().item())
    print("μ mean:", mu.mean().item(), "σ mean:", sigma.mean().item())

    # --------------------------------------------------------------------- #
    # 3.  Spectrogram ➜ (μ,σ)-norm ➜ AE ➜ denorm                            #
    # --------------------------------------------------------------------- #
    print("🔄 Computing spectrograms (log-mag + phase)…")
    specs_np = compute_complex_spectrogram(
        segments,
        fs       = args.fs,
        nperseg  = args.nperseg,
        noverlap = args.noverlap,
    )                                              # (N, F, T, 2C)

    specs_cf = specs_np.transpose(0, 3, 1, 2)      # (N, 2C, F, T)
    # ----- split & stack like training ---------------------------------
    C        = specs_cf.shape[1] // 2
    mag_log  = specs_cf[:, 0::2]                    # log|S|
    phase    = specs_cf[:, 1::2]                    # φ
    specs_cf = np.concatenate(
        [mag_log,
         np.sin(phase),
         np.cos(phase)], axis=1                     # (N,3C,F,T)
    ).astype(np.float32)
    specs_t  = torch.from_numpy(specs_cf).float().to(device)

    # normalise exactly as during training ---------------------------------
    specs_norm = specs_t.clone()
    specs_norm[:, :C] = (specs_norm[:, :C] - mu) / sigma

    # encode→decode --------------------------------------------------------
    with torch.no_grad():
        recon_norm, _ = specAE(specs_norm)         # still Z-space-normalized
        recon_cf = recon_norm.clone()
        recon_cf[:, :C] = recon_cf[:, :C] * sigma + mu # back to real log-mag / phase

    print("🔍 recon_cf stats → mean:", recon_cf.mean().item(), "std:", recon_cf.std().item())
    recon_np = recon_cf.cpu().numpy().transpose(0, 2, 3, 1)

    print("recon_cf shape:", recon_cf.shape)  # should be (N, 2C, F, T)
    print("segments shape:", segments.shape)  # should be (N, T, C)

    print("🔍 enc_mag.dw_freq norm:", specAE.enc_mag.dw_freq.weight.norm().item())

    print("Log-mag mean:", recon_cf[:, 0::2].mean().item())
    print("Log-mag std:",  recon_cf[:, 0::2].std().item())
    print("recon_norm.mean():", recon_norm.mean().item())
    print("recon_norm.std():",  recon_norm.std().item())


    # --------------------------------------------------------------------- #
    # 4.  ISTFT  (torch-Native via inverse_spectrogram helper)              #
    # --------------------------------------------------------------------- #
    
    eps = 1e-5                           # the same epsilon used in compute_complex_spectrogram()
    B, _, F, T = specs_t.shape           # (B, 2C, F, T)

    # 1) ── difference in log-magnitude ────────────────────────────────
    mag_orig  = torch.exp(specs_t[:, 0::2]) - eps          # (B, C, F, T)
    mag_recon = torch.exp(recon_cf[:, 0::2]) - eps
    mae_mag   = (mag_orig - mag_recon).abs().mean().item()
    print(f"🔬 |mag| MAE (orig vs recon): {mae_mag:.4f}")

    # ----- phase error ---------------------------------------------------
    sinφ_t, cosφ_t = specs_t[:, C:2*C],  specs_t[:, 2*C:]
    sinφ_r, cosφ_r = recon_cf[:, C:2*C], recon_cf[:, 2*C:]

    phase_orig  = (torch.atan2(sinφ_t, cosφ_t) + torch.pi) % (2*torch.pi)
    phase_recon = (torch.atan2(sinφ_r, cosφ_r) + torch.pi) % (2*torch.pi)

    cos_sim_phase = torch.cos(phase_orig - phase_recon).mean().item()
    print(f"🔬 ⟨cos(Δφ)⟩ = {cos_sim_phase:.3f}   (1.0 = perfect)")


    b, ch = 0, 0
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(phase_orig[b, ch].cpu(), aspect='auto', origin='lower', cmap="twilight")
    ax[0].set_title("Original Phase φ")
    ax[1].imshow(phase_recon[b, ch].cpu(), aspect='auto', origin='lower', cmap="twilight")
    ax[1].set_title("Reconstructed Phase φ")
    plt.tight_layout(); plt.show()

    delta_phi = torch.atan2(torch.sin(phase_recon - phase_orig),
                        torch.cos(phase_recon - phase_orig))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(delta_phi[b, ch].cpu(), aspect='auto', origin='lower', cmap="twilight_shifted")
    ax.set_title("Wrapped Δφ (Recon − Orig)")
    plt.tight_layout(); plt.show()

    plt.hist(phase_recon.flatten().cpu().numpy(), bins=100)
    plt.title("Histogram of reconstructed φ"); plt.show()

    # Add after phase_orig, phase_recon
    if_orig = phase_orig[..., 1:] - phase_orig[..., :-1]
    if_recon = phase_recon[..., 1:] - phase_recon[..., :-1]

    plt.figure(figsize=(10, 4))
    plt.imshow((if_recon - if_orig)[0, 0].cpu(), aspect='auto', origin='lower', cmap="coolwarm")
    plt.colorbar()
    plt.title("Δ IF (Recon − Orig)")
    plt.show()

    cos_dist = torch.cos(phase_orig - phase_recon)
    plt.figure(figsize=(10, 4))
    plt.imshow(cos_dist[0, 0].cpu(), aspect='auto', origin='lower', cmap="coolwarm")
    plt.colorbar()
    plt.title("Cosine Similarity: cos(φ_pred - φ_true)")
    plt.show()

    # 2) ── quick visual for the first channel of the first segment ────
    b, ch = 0, 0
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(mag_orig[b, ch].cpu(), aspect='auto', origin='lower')
    ax[0].set_title("Original |S|")
    ax[1].imshow(mag_recon[b, ch].cpu(), aspect='auto', origin='lower')
    ax[1].set_title("Reconstructed |S|")
    plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    #  Rebuild (N, F, T, 2 C) array expected by inverse_spectrogram
    #      layout: [log|S|, φ, log|S|, φ, …]
    # ------------------------------------------------------------------
    # ---- sanity check -----------------------------------------------------------------------------
    use_perfect = False          # ← flip to True for the test of inverse and overlay plots
    if use_perfect:
        recon_cf[:, :C]  = specs_t[:, :C]      # log|S|
        recon_cf[:, C:2*C] = specs_t[:, C:2*C] # sin
        recon_cf[:, 2*C:]  = specs_t[:, 2*C:]  # cos
    # -----------------------------------------------------------------------------------------------

    mag_log = recon_cf[:, :C].clone()           # (B, C, F, T)      already log|S|
    mag_log = torch.clamp(mag_log, min=-10.0, max=10.0)   # ← add this line
    sinφ    = recon_cf[:, C:2*C]
    cosφ    = recon_cf[:, 2*C:]

    phase   = torch.atan2(sinφ, cosφ)   # (B, C, F, T)      radians  –π..π

    # interleave log|S| and phase along channel axis
    spec_lp = torch.empty((mag_log.size(0), 2*C, *mag_log.shape[2:]),
                        dtype=mag_log.dtype, device=mag_log.device)
    spec_lp[:, 0::2] = mag_log          # even channels  → log|S|
    spec_lp[:, 1::2] = phase            # odd  channels  → φ

    # (B, 2C, F, T)  →  (B, F, T, 2C)  for inverse_spectrogram()
    spec_lp = spec_lp.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

    # ---------------- ISTFT -------------------------------------------
    print("🔄 ISTFT reconstruction…")
    recon_ts = inverse_spectrogram(
        spec_lp,
        time_length = segments.shape[1],
        fs          = args.fs,
        nperseg     = args.nperseg,
        noverlap    = args.noverlap,
    )                                   # (B, T, C = 12)


    rms_recon = recon_ts.std(axis=1).mean()
    print(f"⚡  RMS of waveform after ISTFT: {rms_recon:.4f}")
    # --------------------------------------------------------------------- #
    # 5.  Align recon ↔ original via x-corr, compute metrics, plot          #
    # --------------------------------------------------------------------- #
    print("🔧 Aligning reconstructions via x-corr…")
    aligned, lags = [], []
    for seg, rec in zip(segments, recon_ts):          # seg,rec are (T,C)
        ali, lag = align_by_xcorr(seg, rec)
        aligned.append(ali); lags.append(lag)
    aligned = np.stack(aligned)

    print("📊 Metrics …")
    metrics = calculate_reconstruction_metrics(segments, aligned)
    print(f"✅ RMSE: {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
    print(f"✅ NCC:  {metrics['ncc_mean']:.4f} ± {metrics['ncc_std']:.4f}")

    # --------------------------------------------------------------------- #
    # 6.  Persist & visualise                                               #
    # --------------------------------------------------------------------- #
    os.makedirs("reconstruction_results", exist_ok=True)
    pd.DataFrame({
        "Segment":       range(args.n_segments),
        "Test ID":       [m["test_id"]      for m in metadata],
        "Trace Index":   [m["trace_index"]  for m in metadata],
        "Peak Position": [m["peak_position"]for m in metadata],
        "Lag":           lags,
        "RMSE":          metrics["rmse_per_segment"],
        "NCC":           metrics["ncc_per_segment"],
    }).to_csv("reconstruction_results/ae_metrics.csv", index=False)
    print("📁 Saved detailed metrics ➜ reconstruction_results/ae_metrics.csv")

    print("📈 Overlay plots …")
    # ------------------------------------------------------------------ 8 --
    #  Show overlays (time-series)                                        #
    # --------------------------------------------------------------------- #
    for i in range(args.n_segments):
        plot_overlay_all_channels(
            original      = segments,      # (N,T,C)
            reconstructed = aligned,       # (N,T,C)
            segment_idx   = i,
            fs            = args.fs,
            title_extra   = f"segment {i} – After AE"
        )

    print("✅ Done – originals vs AE reconstructions plotted & scored.")


if __name__ == "__main__":
    main()
