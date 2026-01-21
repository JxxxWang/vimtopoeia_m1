"""Merge multiple HDF5 dataset files into one."""

import click
import h5py
import hdf5plugin
import numpy as np
from pathlib import Path
from tqdm import tqdm


@click.command()
@click.argument("output_file", type=str)
@click.argument("input_files", type=str, nargs=-1, required=True)
def merge_datasets(output_file: str, input_files: tuple[str]):
    """Merge multiple HDF5 dataset files into OUTPUT_FILE.
    
    Usage:
        python scripts/merge_datasets.py data/merged.h5 data/part1.h5 data/part2.h5 data/part3.h5
    """
    
    print(f"Merging {len(input_files)} files into {output_file}")
    
    # First, get total number of samples and verify consistency
    total_samples = 0
    sample_info = None
    
    for input_file in input_files:
        with h5py.File(input_file, "r") as f:
            num_samples = f["audio"].shape[0]
            total_samples += num_samples
            
            if sample_info is None:
                sample_info = {
                    "channels": f["audio"].shape[1],
                    "audio_length": f["audio"].shape[2],
                    "mel_shape": f["mel_spec"].shape[1:],
                    "num_params": f["param_array"].shape[1],
                    "attrs": dict(f["audio"].attrs),
                }
                print(f"Dataset info: {sample_info}")
            
            print(f"{input_file}: {num_samples} samples")
    
    print(f"\nTotal samples: {total_samples}")
    
    # Create output file with merged datasets
    with h5py.File(output_file, "w") as out_f:
        # Create datasets
        audio_dataset = out_f.create_dataset(
            "audio",
            shape=(total_samples, sample_info["channels"], sample_info["audio_length"]),
            dtype=np.float16,
            compression=hdf5plugin.Blosc2(),
        )
        
        mel_dataset = out_f.create_dataset(
            "mel_spec",
            shape=(total_samples, *sample_info["mel_shape"]),
            dtype=np.float32,
            compression=hdf5plugin.Blosc2(),
        )
        
        param_dataset = out_f.create_dataset(
            "param_array",
            shape=(total_samples, sample_info["num_params"]),
            dtype=np.float32,
            compression=hdf5plugin.Blosc2(),
        )
        
        # Copy attributes
        for key, value in sample_info["attrs"].items():
            audio_dataset.attrs[key] = value
        
        # Copy data from each input file
        current_idx = 0
        for input_file in tqdm(input_files, desc="Merging files"):
            with h5py.File(input_file, "r") as in_f:
                num_samples = in_f["audio"].shape[0]
                
                # Copy in batches to avoid memory issues
                batch_size = 1000
                for start in range(0, num_samples, batch_size):
                    end = min(start + batch_size, num_samples)
                    
                    audio_dataset[current_idx:current_idx + (end-start)] = in_f["audio"][start:end]
                    mel_dataset[current_idx:current_idx + (end-start)] = in_f["mel_spec"][start:end]
                    param_dataset[current_idx:current_idx + (end-start)] = in_f["param_array"][start:end]
                    
                    current_idx += (end - start)
    
    print(f"\nMerged {total_samples} samples into {output_file}")


if __name__ == "__main__":
    merge_datasets()
