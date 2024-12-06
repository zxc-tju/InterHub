import numpy as np
from pathlib import Path

from trajdata.data_structures import Scene
from trajdata.caching import  EnvCache
from trajdata import  VectorMap
from trajdata.data_structures import AgentType
from trajdata.caching.df_cache import DataFrameCache

def load_random_scene(cache_path: Path, env_name: str, scene_dt: float) -> Scene:
    env_cache = EnvCache(cache_path)
    scenes_list = env_cache.load_env_scenes_list(env_name)
    random_scene_name = scenes_list[np.random.randint(0, len(scenes_list))].name
    print(scenes_list)
    print(random_scene_name)

    return env_cache.load_scene(env_name, random_scene_name, scene_dt)


def print_lane_connections(vector_map: VectorMap, lane_id: str):
    # Get the specific lane object
    lane = vector_map.get_road_lane(lane_id)

    # Print upstream lanes
    print("Previous Lanes:")
    for prev_lane_id in lane.prev_lanes:
        print(f"  - {prev_lane_id}")

    # Print downstream lanes
    print("Next Lanes:")
    for next_lane_id in lane.next_lanes:
        print(f"  - {next_lane_id}")

    # Print left adjacent lanes
    print("Adjacent Lanes Left:")
    for left_lane_id in lane.adj_lanes_left:
        print(f"  - {left_lane_id}")

    # Print right adjacent lanes
    print("Adjacent Lanes Right:")
    for right_lane_id in lane.adj_lanes_right:
        print(f"  - {right_lane_id}")

def current_lane_id(lane_kd_tree, query_point, distance_threshold = 3, heading_threshold = 20):  # m, angle   # Use appropriate distance and heading thresholds for querying
    heading_threshold = np.pi /heading_threshold # Heading threshold in radians
    # Get possible lane indices
    lane_indices = lane_kd_tree.current_lane_inds(
        xyzh=query_point,
        distance_threshold=distance_threshold,
        heading_threshold=heading_threshold
    )
    return lane_indices


def get_agent_states(interact_ids, all_agents, vec_map, lane_kd_tree, sc, desired_scene, column_dict, all_timesteps):
    """
    Retrieves the states and lane information for each agent in the given scene.

    Args:
        interact_ids (list): List of agent IDs to focus on for interaction analysis.
        all_agents (list): List of all agents present in the scene.
        vec_map (VectorMap): The vector map of the environment.
        lane_kd_tree: KD-tree for lanes used for proximity searches.
        sc (DataFrameCache): Cache object for accessing scene data.
        desired_scene: Scene object containing details about the scene.
        column_dict (dict): Dictionary mapping column names to their indices in raw state data.
        all_timesteps (list): List of all timesteps available in the scene.

    Returns:
        tuple: A tuple containing:
            - agent_states (np.ndarray): An array with state information for each agent across timesteps.
            - agent_lane_ids (dict): A dictionary with lane IDs assigned to each agent for each timestep.
    """
    # Initialize the states array for all agents (dimensions: num_agents x num_timesteps x 8 state variables)
    agent_states = np.zeros((len(all_agents), desired_scene.length_timesteps, 8))
    
    # Initialize a dictionary to hold lane IDs for each agent at each timestep
    agent_lane_ids = {agent.name: [0] * len(all_timesteps) for agent in desired_scene.agents}

    # Iterate through each agent in the scene
    for agent in desired_scene.agents:
        current_lane = None
        
        # Get indices for state variables (x, y, z, heading) from column_dict
        x_index = column_dict['x']
        y_index = column_dict['y']
        z_index = column_dict['z']
        heading_index = column_dict['heading']

        # Iterate through each timestep for the agent
        for t in range(agent.first_timestep, agent.last_timestep + 1):
            # Retrieve the raw state of the agent at the given timestep
            raw_state = sc.get_raw_state(agent_id=agent.name, scene_ts=t)
            query_point = np.array([raw_state[x_index], raw_state[y_index], raw_state[z_index], raw_state[heading_index]])
            
            # Find lane indices using KD-tree
            lane_indices = current_lane_id(lane_kd_tree, query_point)
            lane_indices = [vec_map.lanes[i].id for i in lane_indices]

            # Determine the most appropriate lane for the agent
            if len(lane_indices) > 1:
                query_point = np.array([raw_state[x_index], raw_state[y_index], raw_state[z_index]])
                closest = lane_kd_tree.closest_polyline_ind(query_point)
                closest_lane_id = vec_map.lanes[int(closest)].id
                if closest_lane_id in lane_indices:
                    chosen_lane = closest_lane_id
                else:
                    chosen_lane = next((lan for lan in lane_indices if lan == current_lane), lane_indices[0])
            elif len(lane_indices) == 0:
                # If no lanes are found, choose the closest lane
                query_point = np.array([raw_state[x_index], raw_state[y_index], raw_state[z_index]])
                closest = lane_kd_tree.closest_polyline_ind(query_point)
                chosen_lane = vec_map.lanes[int(closest)].id
            else:
                # If only one lane is found, select it
                chosen_lane = lane_indices[0]

            current_lane = chosen_lane

            try:
                # Update agent states with raw state data
                agent_index = all_agents.index(agent.name)
                timestep_index = all_timesteps.index(t)
                agent_states[agent_index, timestep_index, :] = raw_state
            except:
                continue

            # Update lane ID for the agent at the given timestep
            agent_lane_ids[agent.name][timestep_index] = chosen_lane

    return agent_states, agent_lane_ids