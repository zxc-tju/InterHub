
import os
import argparse
from trajdata import UnifiedDataset

def main():
    # set the argument parser
    parser = argparse.ArgumentParser(description="Trajectory Data Preprocessing")
    parser.add_argument('--desired_data',
                        default='interaction_multi',
                        help='The dataset you want to unify (default: interaction_multi)')
    parser.add_argument(
        "--env_name",
        default=None,
        help=(
            "Raw trajdata environment name for data_dirs. Defaults to the part "
            "before '-' in desired_data, e.g. av2_motion_forecasting for "
            "av2_motion_forecasting-train."
        ),
    )
    parser.add_argument(
        "--load_path",
        default='data/0_origin_datasets/interaction_multi',
        help="Path to your dataset scenario set (default: 'data/0_origin_datasets/interaction_multi')"
    )
    parser.add_argument(
        "--save_path",
        default='data/1_unified_cache',
        help="Path to save your processed data (default: 'data/1_unified_cache')"
    )
    parser.add_argument(
        "--use_multiprocessing",
        action="store_true",
        help="Use multiprocessing for data processing"
    )
    parser.add_argument(
        "--no_rebuild_cache",
        dest="rebuild_cache",
        action="store_false",
        default=True,
        help="Reuse existing scene cache when possible instead of rebuilding it"
    )
    parser.add_argument(
        "--no_rebuild_maps",
        dest="rebuild_maps",
        action="store_false",
        default=True,
        help="Reuse existing map cache when possible instead of rebuilding maps"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=os.cpu_count(),
        help=f"Number of processes to use if multiprocessing is enabled (default: {os.cpu_count()})"
    )

    args = parser.parse_args()
    env_name = args.env_name or args.desired_data.split("-")[0]

    dataset = UnifiedDataset(
        desired_data=[args.desired_data],
        standardize_data=False,
        rebuild_cache=args.rebuild_cache,
        rebuild_maps=args.rebuild_maps,
        centric="scene",
        num_workers=args.processes if args.use_multiprocessing else 1,
        verbose=True,
        incl_vector_map=True,
        cache_type="dataframe",
        cache_location=args.save_path,
        data_dirs={
            env_name: args.load_path
        },
    )

    print(f"Raw Env Name: {env_name}")
    print(f"Total Data Samples: {len(dataset):,}")

if __name__ == "__main__":
    main()
