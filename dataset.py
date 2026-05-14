import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NiftiDataset(Dataset):
    """Loads .nii / .nii.gz volumes, resizes to isize³ (or isize² for ndim=2),
    and normalises voxel intensities to [-1, 1] per volume."""

    def __init__(self, directory: str, isize: int, ndim: int = 3, normalize: bool = True):
        self.paths = sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        )
        if not self.paths:
            raise ValueError(f"No .nii/.nii.gz files found in {directory!r}")
        self.isize = isize
        self.ndim = ndim
        self.normalize = normalize

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = nib.load(self.paths[idx]).get_fdata(dtype=np.float32)

        if self.ndim == 3:
            if data.ndim == 4:
                data = data[..., 0]  # drop time dim; take first frame
            if data.ndim != 3:
                raise ValueError(f"{self.paths[idx]}: expected 3-D volume, got shape {data.shape}")
        else:
            if data.ndim == 3:
                data = data[:, :, data.shape[2] // 2]  # middle axial slice
            if data.ndim != 2:
                raise ValueError(f"{self.paths[idx]}: expected 2-D slice, got shape {data.shape}")

        tensor = torch.from_numpy(data).unsqueeze(0)  # (1, *spatial)

        # Resize to target isize if needed
        target = (self.isize,) * self.ndim
        if tuple(tensor.shape[1:]) != target:
            mode = "trilinear" if self.ndim == 3 else "bilinear"
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=target,
                mode=mode,
                align_corners=False,
            ).squeeze(0)

        if self.normalize:
            # Step 1 — clamp to the detected range convention:
            #   min >= -0.1  →  data lives in [0, 1]  → clamp then rescale to [-1, 1]
            #   min <  -0.1  →  data lives in [-1, 1] → clamp directly
            if tensor.min().item() >= -0.1:
                tensor = tensor.clamp(0.0, 1.0).mul(2.0).sub(1.0)
            else:
                tensor = tensor.clamp(-1.0, 1.0)

            # Step 2 — min-max correction if output doesn't fully span [-1, 1]
            # e.g. a volume whose raw max is 0.97 would only reach 0.94 after step 1
            lo, hi = tensor.min(), tensor.max()
            if (hi - lo) > 1e-6 and (lo > -0.999 or hi < 0.999):
                tensor = 2.0 * (tensor - lo) / (hi - lo) - 1.0

        return tensor  # (1, isize, isize[, isize])


def check_data_range(dataset: "NiftiDataset", label: str, n_samples: int = 5) -> None:
    """Sample up to `n_samples` volumes and print raw + normalised intensity stats.

    Flags common problems:
      - Flat volume  : raw min == max (empty or constant image)
      - Wrong range  : normalised output not in [-1, 1]
      - Large outlier: raw max > 5000 (likely CT HU without windowing)
    """
    indices = np.linspace(0, len(dataset) - 1, min(n_samples, len(dataset)), dtype=int)

    orig_mins, orig_maxs, orig_means, orig_stds = [], [], [], []
    raw_mins,  raw_maxs,  raw_means,  raw_stds  = [], [], [], []
    out_mins,  out_maxs,  out_means,  out_stds  = [], [], [], []
    warnings = []

    # Original: read NIfTI file directly, no resizing, no normalization
    for i in indices:
        data = nib.load(dataset.paths[i]).get_fdata(dtype=np.float32).ravel()
        orig_mins.append(float(data.min()))
        orig_maxs.append(float(data.max()))
        orig_means.append(float(data.mean()))
        orig_stds.append(float(data.std()))
        if data.max() == data.min():
            warnings.append(f"FLAT volume at index {i}: min==max=={data.min():.4f}")

    # Resized + unnormalised (after interpolation, before normalization)
    orig_norm = dataset.normalize
    dataset.normalize = False
    for i in indices:
        raw = dataset[i].numpy()
        raw_mins.append(float(raw.min()))
        raw_maxs.append(float(raw.max()))
        raw_means.append(float(raw.mean()))
        raw_stds.append(float(raw.std()))
    dataset.normalize = orig_norm

    # Final: resized + normalised
    for i in indices:
        out = dataset[i].numpy()
        out_mins.append(float(out.min()))
        out_maxs.append(float(out.max()))
        out_means.append(float(out.mean()))
        out_stds.append(float(out.std()))

    print(f"\n[data check] {label}  ({len(dataset)} volumes, sampled {len(indices)})")
    print(f"  original min={min(orig_mins):8.3f}  max={max(orig_maxs):8.3f}"
          f"  mean={float(np.mean(orig_means)):8.3f}  std={float(np.mean(orig_stds)):8.3f}")
    print(f"  resized  min={min(raw_mins):8.3f}  max={max(raw_maxs):8.3f}"
          f"  mean={float(np.mean(raw_means)):8.3f}  std={float(np.mean(raw_stds)):8.3f}")
    print(f"  normed   min={min(out_mins):8.4f}  max={max(out_maxs):8.4f}"
          f"  mean={float(np.mean(out_means)):8.4f}  std={float(np.mean(out_stds)):8.4f}"
          f"  (expected [-1, 1])")

    if max(raw_maxs) > 5000:
        warnings.append(f"  LARGE raw max ({max(raw_maxs):.0f}) — CT HU? consider windowing before use.")
    if min(out_mins) < -1.01 or max(out_maxs) > 1.01:
        warnings.append(f"  Normalised range outside [-1,1]: [{min(out_mins):.4f}, {max(out_maxs):.4f}]")

    for w in warnings:
        print(f"  WARNING: {w}")


class MixedSubset(Dataset):
    """Each call to the constructor draws `per_dataset` random samples from
    each dataset in `datasets`, forming a pooled fake distribution for one
    training epoch.  Re-instantiate every epoch to get fresh random draws."""

    def __init__(self, datasets: list, per_dataset: int):
        self.samples: list[tuple] = []
        for ds in datasets:
            n = len(ds)
            idx = torch.randperm(n)[:per_dataset].tolist()
            for i in idx:
                self.samples.append((ds, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ds, i = self.samples[idx]
        return ds[i]
