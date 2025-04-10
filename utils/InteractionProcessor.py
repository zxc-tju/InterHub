import logging
import os
import gc
from tqdm import tqdm
from concurrent.futures import as_completed
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yaml

from trajdata import UnifiedDataset, MapAPI

from utils.AccelerationCalculator import MultiProcess, AccelerationCalculate
from utils.SceneProcessor import SceneProcessor

# Set up the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_config(file_path: str) -> Dict:
    """Load the YAML configuration file.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        dict: Configuration settings as a dictionary.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

class InteractionProcessorTraj:
    """Class to process trajectory interactions based on configurations."""

    def __init__(self, desired_data: str, param: Optional[Any], cache_location: str, save_path: str, num_workers: int):
        """Initialize the processor with the configuration and parameters.

        Args:
            desired_data (str): The dataset name.
            param (Optional[Any]): Duration (in seconds) for the vehicle's future trajectory.
            cache_location (str): Cache location for trajdata.
            save_path (str): Path to save the results of interaction extractions.
            num_workers (int): Number of workers to use.
            config (str): Path to the configuration file.
        """
        # Set configuration and instance variables
        self.desired_data = desired_data
        self.save_path = save_path
        self.timerange = param


        # Initialize internal state variables for data processing
        self.scene_counter = 0
        self.time_dic_output = {}
        self.a_min_dict = {}
        self.multi_a_min_dict = {}
        self.delta_TTCP_dict = {}
        self.results = {}

        # Initialize calculators and processors
        self.calculator = AccelerationCalculate()
        self.multi_processor = MultiProcess()

        # Initialize dataset with configuration settings
        self.dataset = UnifiedDataset(
            desired_data=[self.desired_data],
            standardize_data=False,
            rebuild_cache=False,
            rebuild_maps=False,
            centric="scene",
            verbose=True,
            cache_location=cache_location,
            num_workers=num_workers,
            incl_vector_map=True,
            data_dirs={self.desired_data:''}
        )

        # Set up MapAPI and cache paths
        self.map_api = MapAPI(self.dataset.cache_path)
        self.cache_path = self.dataset.cache_path

    def process(self):
        results_list = []
        scenes_list = list(self.dataset.scenes())  # Convert generator to list
        num_scenes = len(scenes_list)  # Get total number of scenes
        batch_size = 100  # Number of scenes to process per batch

        # Initialize tqdm progress bar
        with tqdm(total=num_scenes, desc="Processing Scenes", unit="scene") as pbar:
            # Submit tasks in batches
            for start_idx in range(0, num_scenes, batch_size):
                end_idx = min(start_idx + batch_size, num_scenes)
                batch_scenes = scenes_list[start_idx:end_idx]

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(self.process_single_scene, idx, desired_scene): idx
                        for idx, desired_scene in enumerate(batch_scenes, start=start_idx)
                    }

                    for future in as_completed(futures):
                        idx = futures[future]
                        # try:
                        scene_results = future.result()
                        results_list.extend(scene_results)
                        # except Exception as e:
                        #     logger.error(f"Error processing scene {idx}: {e}")
                        # finally:
                        pbar.update(1)

        # Save all results to CSV file
        col_name = ["dataset", "scenario_idx", "track_id", "start", "end", 
                    'intensity', 'PET', 'two/multi', 'vehicle_type', 'AV_included',
                    'key_agents', 'path_relation', 'turn_label', 'priority_label', 'path_category', 
                    'pre_int_i', 'post_int_i', 'pre_int_j', 'post_int_j']
        results_df = pd.DataFrame(results_list, columns=col_name)
        os.makedirs(self.save_path, exist_ok=True)
        results_df.to_csv(f'{self.save_path}/results.csv', index=False)

    def process_single_scene(self, idx, desired_scene) -> List[List]:
        """Process a single scene and return interaction rsesults.

        Args:
            idx (int): Index of the scene.
            desired_scene (Any): The scene data to process.

        Returns:
            List[List]: Processed results for the scene.
        """
        # print(idx)
        # Initialize SceneProcessor for the given scene
        scene_processor = SceneProcessor(
            self.desired_data, idx, desired_scene, self.map_api, self.cache_path,
            self.timerange
        )

        # Process the scene and collect results
        scene_results = scene_processor.process_scene(
            self.time_dic_output, self.a_min_dict, self.multi_a_min_dict, self.delta_TTCP_dict, self.results
        )

        del scene_processor, desired_scene
        gc.collect()

        return scene_results



