"""
End-to-end evaluation of the **Multi-Modal Latent Diffusion** model.

Includes
* Encoder/decoder reconstruction metrics
* Diffusion sampling (optional) + latent-FID
* Quick qualitative plots

Run:
    python -m diff_pt.tests.test_diffusion \
           --ckpt_dir results_diff \
           --batch 32 \
           --samples 16
"""

import argparse, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from diff_pt import model as dm, io, vis
from .utils import load_train_stats, spectrograms_to_timeseries
from diff_pt.tests.test_reconstruction import _parse as _base_parse    # reuse flags
from .utils import spectrograms_to_timeseries as s2ts

# --------------------------------------------------------------------------- #
# CLI (extends base flags)                                                    #
# --------------------------------------------------------------------------- #
def _parse():
    p = _base_parse()
    p.add_argument("--samples", type=int, default=8,
                   help="Nr. of unconditional samples for latent-FID")
    p.add_argument("--no_diffusion", action="store_true",
                   help="Skip diffusion sampling; only do reconstruction.")
    return p.parse_args()

# --------------------------------------------------------------------------- #
# MAIN                                                                        #
# --------------------------------------------------------------------------- #
def main():
    args   = _parse()
    device = torch.device(args.device)

    ckpt_dir   = Path(args.ckpt_dir)
    stats_path = ckpt_dir / "train_stats.npy"
    mu, sigma  = load_train_stats(stats_path)

    # ---------- cached data -------------------------------------------------
    (accel_dict, _, heats,
     segments, specs, ids, *_ ) = io.loader.load_data(recompute=False)

    specs = specs.transpose(0, 3, 1, 2)
    masks = np.stack([heats[i] for i in ids], 0).transpose(0, 3, 1, 2)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(specs).float(),
                      torch.from_numpy(masks).float(),
                      torch.from_numpy(ids)),
        batch_size=args.batch, shuffle=False, num_workers=4)

    # ---------- rebuild full model ------------------------------------------
    C = specs.shape[1] // 2
    F, T = specs.shape[2:]
    Hm, Wm = masks.shape[2:]

    specAE = dm.SpectrogramAutoencoder(256, C, F, T).to(device)
    maskAE = dm.MaskAutoencoder(256, (Hm, Wm)).to(device)
    mld    = dm.MultiModalLatentDiffusion(specAE, maskAE,
                                          latent_dim=256,
                                          modality_names=["spec", "mask"],
                                          device=device)

    io.load_autoencoders(mld.autoencoders, device,
                         load_dir=ckpt_dir / "autoencoders")
    io.load_diffusion_model(mld.diffusion_model,
                            device,
                            path=ckpt_dir / "diffusion" / "final_diffusion_model.pt")

    # ---------- reconstruction metrics --------------------------------------
    rec_spec, rec_mask, dices, ious = [], [], [], []
    with torch.no_grad():
        for spec, mask, _ in loader:
            spec, mask = spec.to(device), mask.to(device)
            r_s, _ = specAE(spec)
            r_m, _ = maskAE(mask)

            rec_spec.append(dm.losses.complex_spectrogram_loss(spec, r_s).item())
            rec_mask.append(dm.losses.custom_mask_loss(mask, r_m).item())

            d, i = dm.losses.dice_iou_scores(mask, r_m)
            dices.extend(d); ious.extend(i)

    print(f"🔹 recon-spec loss : {np.mean(rec_spec):.4f}")
    print(f"🔹 recon-mask loss : {np.mean(rec_mask):.4f}")
    print(f"🔹 dice            : {np.mean(dices):.4f}")
    print(f"🔹 IoU             : {np.mean(ious) :.4f}")

    # ---------- optional diffusion sampling ---------------------------------
    if args.no_diffusion:
        print("⏩  Skipping diffusion sampling.")
        return

    with torch.no_grad():
        fake = mld.sample(batch_size=args.samples)

    # latent-FID -------------------------------------------------------------
    z_real, z_fake = [], []
    with torch.no_grad():
        for spec, mask, _ in loader:
            spec, mask = spec.to(device), mask.to(device)
            z_s, _ = specAE.encoder(spec), maskAE.encoder(mask)
            z_real.append(torch.cat([z_s, z_m], 1).cpu())

    for i in range(args.samples):
        z_s  = specAE.encoder(fake["spec"][i:i+1].to(device))
        z_m  = maskAE.encoder(fake["mask"][i:i+1].to(device))
        z_fake.append(torch.cat([z_s, z_m], 1).cpu())

    z_real = torch.cat(z_real).numpy()
    z_fake = torch.cat(z_fake).numpy()

    fid = dm.losses.latent_fid(z_real, z_fake)
    print(f"🔹 latent-FID      : {fid:.4f}")

    # ---------- quick qualitative visual -----------------------------------
    rec_batch = fake["spec"][:4].cpu()
    ts = s2ts(rec_batch, mu=mu, sigma=sigma, hop_align=True)
    print("👀  quick-check ISTFT of first 4 diffusion samples:", ts.shape)


if __name__ == "__main__":
    main()
