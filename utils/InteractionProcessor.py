import logging
import os
import gc
from tqdm import tqdm
from itertools import islice
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

RESULT_COLUMNS = [
    "dataset", "folder", "scenario_idx", "track_id", "start", "end",
    "intensity", "PET", "two/multi", "vehicle_type", "AV_included",
    "key_agents", "pre_int_i", "post_int_i", "pre_int_j", "post_int_j",
    "path_category", "path_relation", "turn_label", "priority_label",
    "original_data_file", "original_scene_id", "original_track_id",
]

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

    def __init__(
        self,
        desired_data: str,
        param: Optional[Any],
        cache_location: str,
        save_path: str,
        num_workers: int,
        output_file: str = "results.csv",
        start_scene: int = 0,
        max_scenes: Optional[int] = None,
        batch_size: int = 100,
        append: bool = False,
    ):
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
        self.output_file = output_file
        self.start_scene = max(0, start_scene)
        self.max_scenes = max_scenes
        self.batch_size = max(1, batch_size)
        self.num_workers = max(1, num_workers)
        self.append = append


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
            data_dirs={self.desired_data: ''}
        )

        # Set up MapAPI and cache paths
        self.map_api = MapAPI(self.dataset.cache_path)
        self.cache_path = self.dataset.cache_path

    def process(self):
        os.makedirs(self.save_path, exist_ok=True)
        output_path = os.path.join(self.save_path, self.output_file)
        write_header = not self.append or not os.path.exists(output_path)
        if not self.append and os.path.exists(output_path):
            os.remove(output_path)
        total_rows = 0
        progress_total = self.max_scenes

        # Initialize tqdm progress bar
        with tqdm(total=progress_total, desc="Processing Scenes", unit="scene") as pbar:
            # Submit tasks in batches
            for batch_start_idx, batch_scenes in self.iter_scene_batches():
                batch_results = []

                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = {
                        executor.submit(self.process_single_scene, idx, desired_scene): idx
                        for idx, desired_scene in enumerate(batch_scenes, start=batch_start_idx)
                    }

                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            scene_results = future.result()
                            batch_results.extend(scene_results)
                        except Exception as e:
                            logger.error(f"Error processing scene {idx}: {e}")
                        pbar.update(1)

                if batch_results:
                    results_df = pd.DataFrame(batch_results, columns=RESULT_COLUMNS)
                    results_df.to_csv(
                        output_path,
                        mode="a",
                        header=write_header,
                        index=False,
                    )
                    write_header = False
                    total_rows += len(batch_results)

        if write_header:
            pd.DataFrame(columns=RESULT_COLUMNS).to_csv(output_path, index=False)

        logger.info(f"Saved {total_rows} interaction rows to {output_path}")

    def iter_scene_batches(self):
        scenes_iter = self.dataset.scenes()
        if self.start_scene:
            scenes_iter = islice(scenes_iter, self.start_scene, None)

        remaining = self.max_scenes
        next_scene_idx = self.start_scene

        while remaining is None or remaining > 0:
            current_batch_size = self.batch_size if remaining is None else min(self.batch_size, remaining)
            batch_scenes = list(islice(scenes_iter, current_batch_size))
            if not batch_scenes:
                break

            yield next_scene_idx, batch_scenes
            batch_len = len(batch_scenes)
            next_scene_idx += batch_len
            if remaining is not None:
                remaining -= batch_len

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
