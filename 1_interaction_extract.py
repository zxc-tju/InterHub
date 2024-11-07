import os
import argparse
from utils.InteractionProcessor import InteractionProcessorTraj


def main():

    parser = argparse.ArgumentParser(description="Interaction Processor")
    parser.add_argument('--desired_data',default='interaction_multi',
                        help='The dataset name')
    parser.add_argument('--timerange', type=int, default=5,
                        help='Duration (in seconds) to consider for the vehicle\'s future trajectory')
    parser.add_argument('--cache_location', type=str, default='data/1_unified_cache',
                        help='Cache location for trajdata (default: "data/1_unified_cache")')
    parser.add_argument('--save_path', type=str, default='data/2_extracted_results',
                        help='Path to save the results of interaction extractions (default: "data/2_extracted_results")')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help=f'Number of workers to use (default: {os.cpu_count()})')

    args = parser.parse_args()

    processor = InteractionProcessorTraj(
        desired_data=args.desired_data,
        param=args.timerange,
        cache_location=args.cache_location,
        save_path=args.save_path,
        num_workers=args.num_workers
    )

    processor.process()


if __name__ == "__main__":
    main()

#