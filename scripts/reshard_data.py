import h5py
import numpy as np

for i in range(22):
    shard = h5py.File(
        f"/data/home/acw585/data_scratch/surge/surgext_train_{i}.hdf5", "w"
    )
    audio_ds = shard.create_dataset(
        "audio", (100_000, 2, 44100 * 4), dtype=np.float16, compression="gzip"
    )
    mel_ds = shard.create_dataset(
        "mel_spec", (100_000, 2, 128, 401), dtype=np.float32, compression="gzip"
    )
    param_ds = shard.create_dataset(
        "param_array", (100_000, 189), dtype=np.float32, compression="gzip"
    )

    print(f"Starting shard {i}.................")
    for j in range(10):
        f_idx = i * 10 + j
        f = h5py.File(f"/data/home/acw585/data_scratch/surge/shard-{f_idx}.hdf5", "r")

        idx_lo = j * 10_000
        idx_hi = (j + 1) * 10_000
        print(f"Copying audio from {idx_lo} to {idx_hi}")
        audio_ds[idx_lo:idx_hi, :, :] = f["audio"]
        print(f"Copying mel_spec from {idx_lo} to {idx_hi}")
        mel_ds[idx_lo:idx_hi, :, :, :] = f["mel_spec"]
        print(f"Copying param_array from {idx_lo} to {idx_hi}")
        param_ds[idx_lo:idx_hi, :] = f["param_array"]
