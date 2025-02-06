#!/usr/bin/env python
import os
from pathlib import Path
import click
import h5py

@click.command()
@click.argument("data_dir", type=str)
@click.option("--pattern", "-p", type=str, default="shard-*.h5",
              help="Glob pattern to find shard files.")
@click.option("--backup/--no-backup", default=True,
              help="Whether to keep a backup of the original file (with a .bak.h5 extension).")
def main(data_dir, pattern, backup):
    """
    Rewrite each HDF5 file matching the given pattern in DATA_DIR so that
    it is created with libver="latest" (i.e. with a superblock version >= 3).
    This is needed for SWMR mode.
    """
    data_dir = Path(data_dir)
    shard_files = list(data_dir.glob(pattern))
    if not shard_files:
        click.echo(f"No files found matching pattern '{pattern}' in {data_dir}.")
        return

    for shard in shard_files:
        click.echo(f"Rewriting {shard} with libver='latest' ...")
        # Create a temporary file (in the same directory) for the rewritten shard.
        temp_file = shard.with_suffix(".temp.h5")
        try:
            with h5py.File(shard, "r") as f_in, \
                 h5py.File(temp_file, "w", libver="latest") as f_out:
                # Copy each top-level item (dataset or group) from the original file.
                for key in f_in:
                    f_in.copy(key, f_out)
            # Optionally, save a backup of the original file.
            if backup:
                backup_file = shard.with_suffix(".bak.h5")
                os.replace(shard, backup_file)
                click.echo(f"Backup saved as {backup_file}")
            else:
                os.remove(shard)
            # Replace the original file with the new one.
            os.replace(temp_file, shard)
            click.echo(f"Replaced {shard}")
        except Exception as e:
            click.echo(f"Error rewriting {shard}: {e}")
            if temp_file.exists():
                temp_file.unlink()

if __name__ == "__main__":
    main()
