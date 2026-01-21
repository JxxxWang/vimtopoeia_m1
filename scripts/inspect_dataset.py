"""Inspect and validate an HDF5 dataset file."""

import click
import h5py
import numpy as np
import hdf5plugin


@click.command()
@click.argument("data_file", type=str)
@click.option("--show-samples", "-s", type=int, default=3, help="Number of random samples to display")
def inspect_dataset(data_file: str, show_samples: int):
    """Inspect and validate an HDF5 dataset file.
    
    Usage:
        python scripts/inspect_dataset.py data/surge_train_40.h5
    """
    
    print(f"Inspecting: {data_file}\n")
    
    with h5py.File(data_file, "r") as f:
        # Show datasets
        print("📊 Datasets:")
        for key in f.keys():
            dataset = f[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        print("\n📋 Attributes:")
        for key, value in f["audio"].attrs.items():
            print(f"  {key}: {value}")
        
        # Validate data
        print("\n✅ Validation:")
        
        # Check audio
        audio = f["audio"]
        num_samples = audio.shape[0]
        print(f"  Total samples: {num_samples}")
        
        # Check for zeros
        num_zero_samples = 0
        for i in range(num_samples):
            if np.all(audio[i] == 0):
                num_zero_samples += 1
        
        if num_zero_samples > 0:
            print(f"  ⚠️  WARNING: {num_zero_samples} empty audio samples found!")
        else:
            print(f"  ✓ No empty audio samples")
        
        # Check mel spectrograms
        mel = f["mel_spec"]
        num_zero_mels = 0
        for i in range(num_samples):
            if np.all(mel[i] == 0):
                num_zero_mels += 1
        
        if num_zero_mels > 0:
            print(f"  ⚠️  WARNING: {num_zero_mels} empty mel spectrograms found!")
        else:
            print(f"  ✓ No empty mel spectrograms")
        
        # Check params
        params = f["param_array"]
        num_zero_params = 0
        for i in range(num_samples):
            if np.all(params[i] == 0):
                num_zero_params += 1
        
        if num_zero_params > 0:
            print(f"  ⚠️  WARNING: {num_zero_params} empty parameter arrays found!")
        else:
            print(f"  ✓ No empty parameter arrays")
        
        # Show random samples
        if show_samples > 0 and num_samples > 0:
            print(f"\n🔍 Random sample inspection:")
            indices = np.random.choice(num_samples, min(show_samples, num_samples), replace=False)
            
            for idx in indices:
                audio_sample = audio[idx]
                mel_sample = mel[idx]
                param_sample = params[idx]
                
                print(f"\n  Sample {idx}:")
                print(f"    Audio: min={audio_sample.min():.4f}, max={audio_sample.max():.4f}, mean={audio_sample.mean():.4f}")
                print(f"    Mel:   min={mel_sample.min():.4f}, max={mel_sample.max():.4f}, mean={mel_sample.mean():.4f}")
                print(f"    Params: min={param_sample.min():.4f}, max={param_sample.max():.4f}, mean={param_sample.mean():.4f}")
                print(f"    Param values: {param_sample[:5]}... (showing first 5)")
        
        print("\n✅ Inspection complete!")


if __name__ == "__main__":
    inspect_dataset()
