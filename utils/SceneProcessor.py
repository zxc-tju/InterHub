import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from shapely.geometry import LineString, Point
from trajdata import MapAPI, VectorMap
from trajdata.data_structures import AgentType
from utils.trajdata_utils import DataFrameCache
from utils.AccelerationCalculator import MultiProcess, AccelerationCalculate
from utils.IntersectionDetector import IntersectionDetector
from utils.labeling_utils import Segment_Labeler


class SceneProcessor:
    """Process individual scenes within the dataset."""
    
    def __init__(self, desired_data: List[str], idx: int, desired_scene: Any, map_api: MapAPI, 
                 cache_path: str, timerange: Optional[int]):
        """Initialize SceneProcessor with scene-specific data and settings.

        Args:
            desired_data (List[str]): Desired data types for processing.
            idx (int): Index of the scene in the dataset.
            desired_scene (Any): The scene data.
            map_api (MapAPI): The map API instance for vector map access.
            cache_path (str): The cache path for storing intermediate data.
            timerange (Optional[int]): Time range for processing.
        """
        self.timerange = timerange

        self.map_api = map_api
        self.desired_data = desired_data
        self.idx = idx
        self.desired_scene = desired_scene
        self.cache_path = cache_path
        self.dt = desired_scene.dt
        
        # Initialize all agents in the scene as a dictionary with vehicle type agents only
        self.all_agents = {agent.name: agent for agent in desired_scene.agents if agent.type == AgentType.VEHICLE}
        self.length_timesteps = desired_scene.length_timesteps
        self.all_timesteps = range(self.length_timesteps)

        # Initialize the cache for the scene and load column information
        self.scene_cache = DataFrameCache(
            cache_path=cache_path,
            scene=desired_scene
        )
        self.column_dict: Dict[str, int] = self.scene_cache.column_dict
        
        # Retrieve vector map associated with the scene
        self.vec_map: VectorMap = self.map_api.get_map(f"{desired_scene.env_name}:{desired_scene.location}")
        self.lane_id = [lane.id for lane in self.vec_map.lanes]

        # Initialize acceleration calculation and multi-process handler
        self.calculator = AccelerationCalculate()
        self.multi_processor = MultiProcess()
        
        # Load k-d trees for spatial indexing in the scene cache
        self.scene_cache.load_kdtrees()

    def process_scene(self, time_dic_output: Dict, a_min_dict: Dict, multi_a_min_dict: Dict, 
                      delta_TTCP_dict: Dict, results: Dict) -> List[List]:
        """Process a specific scene and return interaction results.

        Args:
            time_dic_output (Dict): Output dictionary for timing data.
            a_min_dict (Dict): Dictionary for minimum acceleration data.
            multi_a_min_dict (Dict): Dictionary for multiple acceleration data.
            delta_TTCP_dict (Dict): Dictionary for delta TTCP values.
            results (Dict): Dictionary for storing results.

        Returns:
            List[List]: Processed results for the scene.
        """
        # Initialize agent states and lane information
        self.get_agent_states()

        # Identify moving agents for further processing
        move_agent = self.get_move_agent()
        self.map_processed_data: Dict[int, Dict[str, Dict[str, Any]]] = self.get_fut_line(move_agent)

        # Detect intersections and extract interaction details
        intersectiondetector = IntersectionDetector(
            self.agent_states, self.all_timesteps, self.column_dict, 
            self.all_agents, move_agent, self.map_processed_data
        )

        time_dic = intersectiondetector.intersection_detector()
        pair_list = self.multi_processor.extract_interactions(time_dic)
        intersection_dic = {key: value['intersection_dic'] for key, value in time_dic.items() if key in pair_list}

        # Calculate multi-step accelerations and compile results
        timestamp_msaa, sub_pair_value = self.calculator.get_acceleration(
            pair_list, self.agent_states, self.all_agents, self.column_dict, self.all_timesteps
        )
        scene_results = self.get_results(timestamp_msaa, sub_pair_value, intersection_dic)

        return scene_results

    def get_agent_states(self):
        """Initialize agent states and lane IDs for all agents."""
        x_index, y_index, z_index, heading_index = (
            self.column_dict['x'], self.column_dict['y'], self.column_dict['z'], self.column_dict['heading']
        )

        self.lanes = {}
        self.agent_states = np.zeros((len(self.all_agents), self.length_timesteps, 8 + 1))  # x, y, vx, vy, ax, ay, lane
        self.agent_lane_ids = {}

        for name, agent in self.all_agents.items():
            current_lane = None
            self.agent_lane_ids[name] = [0] * self.length_timesteps

            for t in range(agent.first_timestep, agent.last_timestep + 1):
                raw_state = self.scene_cache.get_raw_state(agent_id=name, scene_ts=t)

                query_point = np.array([
                    raw_state[x_index], raw_state[y_index], raw_state[z_index], raw_state[heading_index]
                ])

                current_lane = self.vec_map.get_closest_lane(xyz=np.array([
                    raw_state[x_index], raw_state[y_index], raw_state[z_index]
                ]))
                current_lane_id = current_lane.id
                self.lanes[current_lane.id] = current_lane

                index_agent = list(self.all_agents.keys()).index(name)
                index_time = self.all_timesteps.index(t)

                # Update the agent_states with raw_state data
                self.agent_states[index_agent, index_time, :-1] = raw_state
                self.agent_states[index_agent, index_time, -1] = current_lane_id
                self.agent_lane_ids[agent.name][index_time] = current_lane_id

    def get_move_agent(self) -> Dict[int, List[str]]:
        """Identify moving agents at each timestamp.

        Returns:
            Dict[int, List[str]]: Dictionary mapping timestamps to lists of active agent IDs.
        """
        move_agent = {}
        for timeindex, timestamp in enumerate(self.all_timesteps):
            # Get all active vehicle IDs at the current timestamp
            current_ids = [
                list(self.all_agents.keys())[index]
                for index, flag in enumerate(self.agent_states[:, timeindex, self.column_dict['vx']])
                if flag != 0
            ]
            move_agent[timestamp] = current_ids
        return move_agent

    def get_fut_line(self, move_agent: Dict[int, List[str]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """Compute future trajectories for each agent.

        Args:
            move_agent (Dict[int, List[str]]): Dictionary of moving agents at each timestamp.

        Returns:
            Dict[int, Dict[str, Dict[str, Any]]]: Nested dictionary with timestamp and agent trajectory data.
        """
        map_processed_data = {}
        for timeindex, timestamp in enumerate(self.all_timesteps):
            map_processed_data[timestamp] = {}

            current_ids = move_agent[timestamp]
            for agent_id in current_ids:
                track_idx = list(self.all_agents.keys()).index(agent_id)
                last_nonzero_index = self.find_last_nonzero_index(
                    self.agent_states, track_idx, [self.column_dict['x'], self.column_dict['y']]
                )
                max_lane_id = self.agent_lane_ids[agent_id][last_nonzero_index]
                max_downstream_lane_ids = self.get_downstream_lane_ids(max_lane_id, self.vec_map, 3)
                lanes = [max_lane_id, max_downstream_lane_ids]

                trajectory, future_time, x, y, complete_lane, speed, distance = self.process_vehicle_tracks(track_idx, timeindex, lanes)

                map_processed_data[timestamp][agent_id] = self.process_tracks(
                    self.agent_states, track_idx, timeindex, self.timerange,
                    [self.column_dict['x'], self.column_dict['y']],
                    [self.column_dict['vx'], self.column_dict['vy']], self.dt,
                    additional_trajectory=trajectory
                )
                
        return map_processed_data
    
    def get_results(self, timestamp_msaa: Dict[int, Dict[Tuple[int, int], List[float]]], sub_pair_value: Dict[Tuple[str, str], List[float]], intersection_dic: Dict[int, Dict[Tuple[str, str], Tuple[float, float]]]) -> List[List]:
        """Process and extract results based on the multi-step acceleration array (MSAA).

        Args:
            timestamp_msaa (Dict[int, Dict[Tuple[int, int], List[float]]]): 
                Dictionary where each key is a timestamp, and the value is another dictionary.
                The inner dictionary contains pairs of agents and their interaction metrics.

        Returns:
            List[List]: A list of lists where each inner list contains processed interaction details 
                        for each pair that meets the criteria.
        """
        acceleration_thresh = 0.01  # Threshold for acceleration values to consider

        # Convert the dictionary to an array representation
        a_min_matrix, row_labels, col_labels = self.dic_to_array(timestamp_msaa)
        result_segments = self.get_segments(row_labels, col_labels, a_min_matrix, acceleration_thresh)
        results = []

        # Iterate over each interaction pair to extract results
        for pair, value in result_segments.items():
            col_dataset = self.desired_data
            col_scenario_idx = self.desired_scene.raw_data_idx
            track_idx = ";".join(str(id) for id in pair)
            start_time = value[0][0]
            end_time = value[-1][-1]
            
            # Check if the duration of the interaction is greater than 3 timesteps
            if end_time - start_time > 3:
                # Extract maximum intensity and minimum PET (Post-Encroachment Time)
                msaa_list = [timestamp_msaa[ts].get(pair, [0, None, None, None])[0] for ts in range(start_time, end_time + 1)]
                intensity = max(msaa_list)
                mean_pet_list = [timestamp_msaa[ts].get(pair, [None, None, None, 0])[3] for ts in range(start_time, end_time + 1)]
                PET = min(mean_pet_list)

                # Determine if it's a two-vehicle or multi-vehicle interaction
                two_multi = 'two' if len(pair) == 2 else 'multi'
                vehicle_type = ['HV' if agent_id != 'ego' else 'AV' for agent_id in pair]
                AV_included = 'AV' if 'ego' in pair else 'all_HV'


                # get key_agents
                if two_multi == 'two':
                    key_pair = pair
                    key_agents = track_idx
                else:
                    # find key agents
                    msaa_pair_dict = {key:value[start_time:end_time+1] for key, value in sub_pair_value.items() if (key[0] in pair)&(key[1] in pair)}
                    max_values = {k: max(v) for k, v in msaa_pair_dict.items()}
                    key_pair = max(max_values, key=max_values.get)
                    key_agents = ";".join(str(id) for id in key_pair)

                # get path_relation
                intersection_point = intersection_dic.get(end_time, {}).get(key_pair, None)
                if intersection_point is None: 
                    intersection_point = intersection_dic.get(end_time, {}).get(key_pair[::-1], None)
                fut_line1, fut_line2 = self.get_line(end_time, key_pair[0], key_pair[1])
                segment_labeler = Segment_Labeler(self.agent_states, self.all_agents, key_pair, end_time, intersection_point, fut_line1, fut_line2, pair)
                path_relation, pre_int_i, post_int_i, pre_int_j, post_int_j  = segment_labeler.get_path_relation_label()

                # get path_category
                path_category = segment_labeler.get_path_category(path_relation)

                # get turn_label
                turn_label = segment_labeler.get_turn_label()

                # get priority_label
                priority = segment_labeler.get_priority_label(self.lanes)

                # Append the extracted data to the results list
                results.append([
                    col_dataset, col_scenario_idx, track_idx, start_time, 
                    end_time, intensity, PET, two_multi, vehicle_type, AV_included, key_agents, path_relation, turn_label, priority, path_category,
                    pre_int_i, post_int_i, pre_int_j, post_int_j
                ])

        return results

    def dic_to_array(self, msaa_dict: Dict[int, Dict[Tuple[int, int], List[float]]]) -> Tuple[np.ndarray, List, List]:
        """Convert the MSAA dictionary into a 2D array format.

        Args:
            msaa_dict (Dict[int, Dict[Tuple[int, int], List[float]]]): The MSAA data structure.

        Returns:
            Tuple[np.ndarray, List, List]: 
                - 2D numpy array representing the acceleration values over time.
                - List of row labels (interaction pairs).
                - List of column labels (timestamps).
        """
        ts_labels = list(msaa_dict.keys())  # Timestamps
        row_labels = set(ts for inner_dict in msaa_dict.values() for ts in inner_dict.keys())  # Interaction pairs
        col_labels = range(min(ts_labels), max(ts_labels) + 1)

        # Create a matrix to store the acceleration values
        matrix = np.zeros((len(row_labels), len(col_labels)))

        # Fill the matrix with values from msaa_dict
        for col_idx, timestamp in enumerate(col_labels):
            for row_idx, pair in enumerate(row_labels):
                if pair in msaa_dict.get(timestamp, {}):
                    matrix[row_idx, col_idx] = msaa_dict[timestamp][pair][0]

        return matrix, row_labels, col_labels

    def get_segments(self, row_labels: List, col_labels: List, a_min_matrix: np.ndarray, 
                    acceleration_thresh: float) -> Dict:
        """Identify segments where acceleration exceeds the threshold.

        Args:
            row_labels (List): Labels for rows (interaction pairs).
            col_labels (List): Labels for columns (timestamps).
            a_min_matrix (np.ndarray): 2D matrix of acceleration values.
            acceleration_thresh (float): Threshold to identify significant segments.

        Returns:
            Dict: Dictionary mapping each row label (interaction pair) to their respective time segments.
        """
        result_segments = {}

        for i, row_label in enumerate(row_labels):
            start_time_step = None
            end_time_step = None

            # Iterate through each column (timestamp) for the current row (interaction pair)
            for j, acceleration in enumerate(a_min_matrix[i, :]):
                if acceleration >= acceleration_thresh:
                    if start_time_step is None:
                        start_time_step = col_labels[j]
                    end_time_step = col_labels[j]
                elif acceleration < acceleration_thresh and start_time_step is not None:
                    # If the acceleration drops below the threshold, finalize the segment
                    result_segments.setdefault(row_label, []).append((start_time_step, end_time_step))
                    start_time_step = None
                    end_time_step = None

            # Handle any ongoing segment at the end
            if start_time_step is not None:
                result_segments.setdefault(row_label, []).append((start_time_step, end_time_step))

        return result_segments

    def find_last_nonzero_index(self, agent_states: np.ndarray, track_id_index: int, position_index: List[int]) -> Optional[int]:
        """Find the last non-zero index for the given agent's trajectory.

        Args:
            agent_states (np.ndarray): The array of agent states.
            track_id_index (int): The index of the track in the agent_states array.
            position_index (List[int]): List containing position indices to check.

        Returns:
            Optional[int]: The last non-zero index, or None if all values are zero.
        """
        # Extract all time steps for the specified position
        track_positions = agent_states[track_id_index, :, position_index[0]]

        # Create a boolean array indicating non-zero positions
        non_zero = track_positions != 0

        # Find the last non-zero index by reversing and using argmax
        last_nonzero_index = len(non_zero) - 1 - np.argmax(non_zero[::-1])

        # Return None if all values are zero
        if not non_zero.any():
            return None

        return last_nonzero_index

    def process_tracks(self, all_tracks: np.ndarray, track_id_index: int, timestamp_index: int, 
                    timerange: int, position_index: List[int], v_index: List[int], dt: float,
                    additional_trajectory: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process the tracks for a vehicle and return its trajectory and velocity.

        Args:
            all_tracks (np.ndarray): Array of all vehicle tracks.
            track_id_index (int): Index of the track for the vehicle.
            timestamp_index (int): The current timestamp index.
            timerange (int): Time range for processing.
            position_index (List[int]): Indices for position information (x, y).
            v_index (List[int]): Indices for velocity information (vx, vy).
            dt (float): Time step duration.
            additional_trajectory (Optional[np.ndarray]): Additional trajectory points.

        Returns:
            Dict[str, Any]: Dictionary with line (trajectory) and velocity.
        """
        # Ensure timerange and timestamp are integers
        timerange = int(timerange / dt)
        timestamp_index = int(timestamp_index)

        # Extract and filter the track based on the given timerange
        track = all_tracks[track_id_index, timestamp_index:timestamp_index + timerange, position_index]
        valid_indices = np.any(track != 0, axis=0)

        # Filter the track based on valid (non-zero) indices
        track = track[:, valid_indices]

        # Transpose and further filter the track data
        track_transposed = track.transpose()
        valid_indices = np.any(track_transposed != 0, axis=1)
        filtered_track = track_transposed[valid_indices]

        if filtered_track.shape[0] < 2:
            # Return if there are fewer than 2 valid points
            return {'line': None, 'velocity': 0}

        # Create a LineString object from the filtered track points
        line = LineString(filtered_track)

        # Calculate the velocity of the vehicle
        v_x, v_y = all_tracks[track_id_index, timestamp_index, v_index]
        v = np.sqrt(v_x ** 2 + v_y ** 2)

        return {'line': line, 'velocity': v}

    def concatenate_trajectory(self, original_line: LineString, additional_trajectory: np.ndarray) -> LineString:
        """Concatenate the original line with additional trajectory points.

        Args:
            original_line (LineString): The original trajectory line.
            additional_trajectory (np.ndarray): Additional trajectory points.

        Returns:
            LineString: The concatenated LineString.
        """
        if additional_trajectory.shape[0] == 0:
            return LineString(additional_trajectory)
        else:
            # Combine the original line and additional trajectory points into one LineString
            additional_trajectory = additional_trajectory[:, :2]  # Take only the first two dimensions (x, y)
            original_coords = np.array(original_line.coords)
            additional_coords = np.array(additional_trajectory)
            concatenated_coords = np.vstack([original_coords, additional_coords])
            return LineString(concatenated_coords)

    def get_downstream_lane_ids(self, lane_id: int, vec_map: VectorMap, max_upstream_lanes: int = 3) -> List[int]:
        """Retrieve downstream lane IDs up to a maximum number of lanes.

        Args:
            lane_id (int): The current lane ID.
            vec_map (VectorMap): The vector map instance.
            max_upstream_lanes (int): Maximum number of downstream lanes to retrieve.

        Returns:
            List[int]: List of downstream lane IDs.
        """
        downstream_lane_ids = []
        current_lane_id = lane_id

        # Traverse and collect downstream lanes up to the maximum specified
        while len(downstream_lane_ids) < max_upstream_lanes:
            try:
                lane = vec_map.get_road_lane(current_lane_id)
            except:
                break
            if not lane.next_lanes:
                break  # Stop if no more downstream lanes
            current_lane_id = next(iter(lane.next_lanes))
            downstream_lane_ids.append(current_lane_id)

        return downstream_lane_ids

    def get_complete_lane(self, x: float, y: float, h: float, center_lane_id: int, 
                        downstream_lane_ids: List[int], vec_map: VectorMap, lane_ids: List[int]) -> np.ndarray:
        """Construct the complete lane by concatenating center and downstream lanes.

        Args:
            x (float): X coordinate of the initial point.
            y (float): Y coordinate of the initial point.
            h (float): Heading angle.
            center_lane_id (int): The center lane ID.
            downstream_lane_ids (List[int]): List of downstream lane IDs.
            vec_map (VectorMap): The vector map instance.
            lane_ids (List[int]): List of all lane IDs.

        Returns:
            np.ndarray: Array of concatenated lane points.
        """
        center_lane = vec_map.get_road_lane(center_lane_id).center
        lane_points = [np.array([[x, y, 0, h]])]
        for lane_id in downstream_lane_ids:
            if lane_id in lane_ids:
                downstream_lane = vec_map.get_road_lane(lane_id).center
                if downstream_lane:
                    lane_points.append(downstream_lane.points)
        return np.vstack(lane_points)

    def extract_path_from_current_position(self, x: float, y: float, lane_points: np.ndarray, distance: float) -> np.ndarray:
        """Extract the path starting from the current position up to a given distance.

        Args:
            x (float): X coordinate of the current position.
            y (float): Y coordinate of the current position.
            lane_points (np.ndarray): Array of lane points.
            distance (float): The distance to extract.

        Returns:
            np.ndarray: Array of extracted points along the path.
        """
        if len(lane_points) == 0:
            return np.array([])  # Return empty array if no lane points provided

        extracted_points = []
        accumulated_distance = 0.0

        for i in range(len(lane_points) - 1):
            point1 = lane_points[i][:2]
            point2 = lane_points[i + 1][:2]

            segment_distance = np.linalg.norm(point2 - point1)

            if accumulated_distance + segment_distance >= distance:
                remaining_distance = distance - accumulated_distance
                ratio = remaining_distance / segment_distance
                interpolated_point = point1 + ratio * (point2 - point1)
                extracted_points.append(np.hstack((interpolated_point, lane_points[i][2:])))
                break

            extracted_points.append(lane_points[i])
            accumulated_distance += segment_distance

        return np.array(extracted_points)


    def process_vehicle_tracks(self, track_id_index: int, timestamp_index: int, lanes: List) -> Tuple:
        """Process the vehicle's track to generate future trajectory and related information.

        Args:
            track_id_index (int): Index of the specific track for the vehicle being processed.
            timestamp_index (int): The current timestamp index within the tracks.
            lanes (List): List containing the current lane and downstream lanes for the vehicle. 
                        The first element is the current lane ID, and the second element is a list of downstream lane IDs.

        Returns:
            Tuple: A tuple containing:
                - trajectory (List): Future trajectory points.
                - future_time (float): The computed future time based on speed.
                - x (float): X-coordinate of the vehicle's current position.
                - y (float): Y-coordinate of the vehicle's current position.
                - complete_lane (List): The complete lane composed of current and downstream lanes.
                - speed (float): The speed of the vehicle at the current timestamp.
                - distance (float): The computed distance the vehicle will travel in the future time.
        """

        # Use self to access all_tracks, column_dict, and other attributes
        all_tracks = self.agent_states
        column_dict = self.column_dict
        time_stamp_list = self.all_timesteps
        vec_map = self.vec_map
        lanes = lanes
        lane_id = self.lane_id

        # Extract the current x, y coordinates, velocities (vx, vy), and heading angle (h)
        x, y, vx, vy, h = (
            all_tracks[track_id_index, time_stamp_list.index(max(time_stamp_list)), column_dict['x']],
            all_tracks[track_id_index, time_stamp_list.index(max(time_stamp_list)), column_dict['y']],
            all_tracks[track_id_index, time_stamp_list.index(max(time_stamp_list)), column_dict['vx']],
            all_tracks[track_id_index, time_stamp_list.index(max(time_stamp_list)), column_dict['vy']],
            all_tracks[track_id_index, time_stamp_list.index(max(time_stamp_list)), column_dict['heading']]
        )
        
        # Calculate the speed of the vehicle based on its velocity components
        speed = np.sqrt(vx ** 2 + vy ** 2)

        # Calculate the future time based on the pre-defined future duration
        future_time = self.calculate_future_time(timestamp_index)
        
        # If the calculated future time is zero, return default values
        if future_time == 0:
            return [], future_time, x, y, [], speed, 0

        # Calculate the distance the vehicle will travel based on speed and future time
        distance = speed * future_time

        complete_lane = []
        trajectory = []

        # Check if downstream lanes are available and compute the complete lane and trajectory
        if len(lanes[1]):
            complete_lane = self.get_complete_lane(x, y, h, lanes[0], lanes[1], vec_map, lane_id)
            trajectory = self.extract_path_from_current_position(x, y, complete_lane, distance)

        # If the trajectory has only one point, it is insufficient and reset it to an empty list
        if len(trajectory) == 1:
            trajectory = []

        return trajectory, future_time, x, y, complete_lane, speed, distance

    def calculate_future_time(self, timestamp: int) -> float:
        """Calculate the future time duration based on the remaining time steps and pre-defined future duration.

        Args:
            timestamp (int): The current timestamp index within the track.

        Returns:
            float: The computed future time, ensuring it does not exceed the remaining time steps.
        """
        # Use self to access class attributes for time_stamp_list, dt
        time_stamp_list = self.all_timesteps
        dt = self.dt

        # Calculate the remaining time from the current timestamp to the last timestamp in the list
        remaining_time = max(time_stamp_list) - timestamp

        # Calculate the future time, ensuring it is non-negative
        future_time = max(0, self.timerange - remaining_time * dt)

        return future_time
    
    def get_line(self, timestamp, track_id1, track_id2):
        """
        get trajectory line for two vehicles at a specific timestamp
        """
        line1 = self.map_processed_data[timestamp].get(track_id1, {}).get('line', None)
        line2 = self.map_processed_data[timestamp].get(track_id2, {}).get('line', None)
        return line1, line2

