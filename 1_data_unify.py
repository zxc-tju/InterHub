
import os
import argparse
from trajdata import UnifiedDataset

def main():
    # set the argument parser
    parser = argparse.ArgumentParser(description="Trajectory Data Preprocessing")
    parser.add_argument('--desired_data',
                        default='lyft_train_full',
                        help='The dataset you want to unify (default: lyft_train_full)')
    parser.add_argument(
        "--load_path",
        default='dataset/lyft/scenes/train_full.zarr',
        help="Path to your dataset scenario set (default: 'dataset/lyft/scenes/train_full.zarr')"
    )
    parser.add_argument(
        "--save_path",
        default='Rawdataset/lyft',
        help="Path to save your processed data (default: 'Rawdataset/lyft')"
    )
    parser.add_argument(
        "--use_multiprocessing",
        action="store_true",
        help="Use multiprocessing for data processing"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=os.cpu_count(),
        help=f"Number of processes to use if multiprocessing is enabled (default: {os.cpu_count()})"
    )

    args = parser.parse_args()

    dataset = UnifiedDataset(
        desired_data=["lyft_train_full"],
        standardize_data=False,
        rebuild_cache=True,
        rebuild_maps=True,
        centric="scene",
        num_workers=args.processes if args.use_multiprocessing else 1,
        verbose=True,
        incl_vector_map=True,
        cache_type="dataframe",
        cache_location=args.save_path,
        data_dirs={
            "lyft_train_full": args.load_path
        },
    )

    print(f"Total Data Samples: {len(dataset):,}")

if __name__ == "__main__":
    main()
