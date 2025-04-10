import os
import math
import pandas as pd
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
from shapely.geometry import box
from scipy.interpolate import interp1d
from shapely.geometry import LineString, Point

from trajdata import MapAPI

from utils.trajdata_utils import DataFrameCache, get_agent_states
# from extract_utils import *



def extend_line(line, distance):
    """
    Extends a line to a specified length if the original line is shorter.

    Args:
        line (LineString): A LineString object representing the trajectory or path.
        distance (float): The desired length to extend the line to.

    Returns:
        LineString: A new LineString object with the extended length.
    """
    # Extract the coordinates of the line and define the start and end points of the last segment
    coords = list(line.coords)
    start, end = Point(coords[-2]), Point(coords[-1])

    # Calculate the direction vector (dx, dy) of the last segment
    dx = end.x - start.x
    dy = end.y - start.y
    segment_length = math.sqrt(dx ** 2 + dy ** 2)

    # Calculate the scaling factor needed to extend the line to the desired distance
    factor = (distance - line.length) / segment_length
    new_x = end.x + dx * factor
    new_y = end.y + dy * factor

    # Add the new point to the coordinates list
    extended_coords = coords + [(new_x, new_y)]

    # Create and return a new LineString with the extended coordinates
    return LineString(extended_coords)

    
def extract_segment_by_distance(line, distance):
    """
    Extracts a segment from the starting point of the LineString up to a specified distance. 
    If the distance exceeds the length of the line, it extends the line to meet the required distance.

    Args:
        line (LineString): The input LineString representing the trajectory.
        distance (float): The desired distance from the start point to extract the segment.

    Returns:
        LineString: A new LineString object representing the extracted or extended segment. 
                    Returns None if the distance is less than or equal to zero.
    """
    if distance <= 0:
        return None  # Return None if the distance is zero or negative

    # Get the total length of the LineString
    total_length = line.length

    # If the distance exceeds the total length, extend the line
    if distance > total_length:
        return extend_line(line, distance)

    # Find the point at the specified distance along the line
    cut_point = line.interpolate(distance)

    # Extract coordinates up to the cut_point
    coords = list(line.coords)
    new_coords = []

    current_length = 0.0
    for i in range(1, len(coords)):
        segment = LineString([coords[i - 1], coords[i]])
        segment_length = segment.length
        if current_length + segment_length > distance:
            # Calculate the split point in the final segment
            remaining_distance = distance - current_length
            split_point = segment.interpolate(remaining_distance)
            new_coords.append((split_point.x, split_point.y))
            break
        new_coords.append(coords[i - 1])
        current_length += segment_length

    new_coords.append((cut_point.x, cut_point.y))

    # Create a new LineString object using the extracted coordinates
    segment = LineString(new_coords)

    return segment

def process_tracks_single(considertime, all_tracks, track_id_index, timestamp_index, timerange, position_index, v_index):
    """
    Processes a single track segment by extracting and extending the trajectory based on velocity and a given time duration.

    Args:
        considertime (float): The time duration used to calculate the segment distance.
        all_tracks (ndarray): The array containing all track information.
        track_id_index (int): The index of the specific track in all_tracks.
        timestamp_index (int): The index of the current timestamp in the track.
        timerange (int): The range of timestamps to consider for this segment.
        position_index (tuple): The indices in all_tracks representing the x and y positions.
        v_index (tuple): The indices in all_tracks representing the x and y velocities.

    Returns:
        dict or None: A dictionary containing the processed 'line' (LineString object), 'velocity' (float), and 'line0' (original line). 
                      Returns None if the track has fewer than 2 valid points.
    """
    # Calculate velocity magnitude
    v_x, v_y = all_tracks[track_id_index, timestamp_index, v_index]
    v = np.sqrt(v_x ** 2 + v_y ** 2)
    distance = v * considertime  # Calculate distance based on velocity and time duration

    # Ensure timerange and timestamp_index are integers
    timerange = int(timerange)
    timestamp_index = int(timestamp_index)

    # Extract and filter the trajectory for the given timerange
    track = all_tracks[track_id_index, timestamp_index:timestamp_index + timerange, position_index]
    valid_indices = np.any(track != 0, axis=0)  # Identify valid (non-zero) points

    # Filter the track using valid indices
    track = track[:, valid_indices]

    # Transpose the track for easier processing and filter again
    track_transposed = track.transpose()
    valid_indices = np.any(track_transposed != 0, axis=1)
    filtered_track = track_transposed[valid_indices]

    # If there are fewer than 2 valid points, return None
    if filtered_track.shape[0] < 2:
        return None

    # Create the original LineString object from the filtered track
    line0 = LineString(filtered_track)
    # Extract a segment of the line up to the calculated distance
    line = extract_segment_by_distance(line0, distance)

    # Return the processed result as a dictionary
    return {
        'line': line,
        'velocity': v,
        'line0': line0,
    }


def is_line_in_view(line_points, view_rect):
    """
    Checks if a line segment intersects with the rectangular view window.

    Args:
        line_points (ndarray): An array of points representing the line segment, typically of shape (N, 2).
        view_rect (shapely.geometry.Polygon): A rectangular view window represented as a Shapely Polygon object.

    Returns:
        bool: True if the line segment intersects with the view rectangle, False otherwise.
    """
    # Create a LineString object from the provided line points
    line = LineString(line_points)

    # Check if the line intersects with the view rectangle
    return line.intersects(view_rect)



def rotate_around_center(pts, center, yaw):
    """
    Rotates a set of points around a given center by a specified angle.

    Args:
        pts (ndarray): An array of shape (N, 2) representing N points (x, y) to be rotated.
        center (ndarray): An array of shape (2,) representing the center (x, y) around which to rotate the points.
        yaw (float): The rotation angle in radians (positive for counter-clockwise rotation).

    Returns:
        ndarray: The rotated points as an array of shape (N, 2).
    """
    # Subtract the center point from all points to shift them to the origin
    shifted_pts = pts - center

    # Create the rotation matrix based on the yaw angle
    rotation_matrix = np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]
    ])

    # Apply the rotation matrix and shift the points back to the original center
    rotated_pts = np.dot(shifted_pts, rotation_matrix) + center

    return rotated_pts

def polygon_xy_from_motionstate(x, y, psi_rad, width=1.6, length=4.0):
    """
    Generates the coordinates of a rectangle (polygon) representing a vehicle based on its position, orientation, and dimensions.

    Args:
        x (float): The x-coordinate of the vehicle's center.
        y (float): The y-coordinate of the vehicle's center.
        psi_rad (float): The orientation angle of the vehicle in radians.
        width (float): The width of the vehicle. Default is 1.6 meters.
        length (float): The length of the vehicle. Default is 4.0 meters.

    Returns:
        ndarray: The coordinates of the vehicle's four corners after rotation based on its orientation.
    """
    # Define the four corners of the vehicle before rotation
    lowleft = (x - length / 2.0, y - width / 2.0)
    lowright = (x + length / 2.0, y - width / 2.0)
    upright = (x + length / 2.0, y + width / 2.0)
    upleft = (x - length / 2.0, y + width / 2.0)

    # Create a numpy array with the coordinates of the corners
    corners = np.array([lowleft, lowright, upright, upleft])

    # Rotate the polygon around the vehicle's center (x, y) by the specified angle (psi_rad)
    rotated_corners = rotate_around_center(corners, np.array([x, y]), yaw=psi_rad - math.pi / 180)

    return rotated_corners


def get_map_and_kdtrees(dataset, desired_scene):
    """
    Retrieves the vector map and KD-trees for the lanes based on the given dataset and scene.

    Args:
        dataset: The dataset object containing information about the scenes and cache paths.
        desired_scene: The scene object for which the map and KD-trees need to be retrieved.

    Returns:
        tuple: A tuple containing:
            - vec_map: The vector map for the given scene's environment and location.
            - lane_kd_tree: The KD-tree for lane information in the scene.
    """
    # Initialize the MapAPI with the cache path from the dataset
    map_api = MapAPI(dataset.cache_path)
    
    # Retrieve the environment name and set up the scene cache
    env_name = desired_scene.env_name
    scene_cache = DataFrameCache(cache_path=dataset.cache_path, scene=desired_scene)
    
    # Load KD-trees for the scene and extract the lane KD-tree
    scene_cache.load_kdtrees()
    lane_kd_tree = scene_cache.get_kdtrees(True)[1]
    
    # Get the vector map for the environment and location specified in the scene
    vec_map = map_api.get_map(f"{env_name}:{desired_scene.location}")
    
    return vec_map, lane_kd_tree



def setup_plot_bounds(interact_ids, mean_time, sc, column_dict, vec_map, all_agents, all_timesteps, dt, considertime):
    """
    Sets up the plot boundaries for visualizing the interaction of agents within the scene.

    Args:
        interact_ids (list): List of IDs for interacting agents.
        mean_time (int): The central timestamp used for determining the plot center.
        sc (DataFrameCache): Cache object for accessing scene data.
        column_dict (dict): Dictionary mapping column names to their indices in raw state data.
        vec_map (VectorMap): The vector map of the environment.
        all_agents (list): List of all agents in the scene.
        all_timesteps (list): List of all timesteps in the scene.
        dt (float): Timestep duration.
        considertime (float): The duration to be considered for plotting.

    Returns:
        tuple: A tuple containing the minimum and maximum bounds for x and y coordinates.
    """
    try:
        # Attempt to get the center coordinates (x, y) based on the first interacting agent at the mean time
        center_x = sc.get_raw_state(agent_id=interact_ids[0], scene_ts=mean_time)[column_dict['x']]
        center_y = sc.get_raw_state(agent_id=interact_ids[0], scene_ts=mean_time)[column_dict['y']]
    except:
        # If the first agent fails, use the last agent in the list
        center_x = sc.get_raw_state(agent_id=interact_ids[-1], scene_ts=mean_time)[column_dict['x']]
        center_y = sc.get_raw_state(agent_id=interact_ids[-1], scene_ts=mean_time)[column_dict['y']]

    # Define the radius for the view area
    r = 100
    x_min, x_max = center_x - r, center_x + r
    y_min, y_max = center_y - r, center_y + r

    view_rect = box(x_min, y_min, x_max, y_max)
    lane_ids = vec_map.lanes

    return x_min, x_max, y_min, y_max


def plot_lanes(plt, vec_map, view_rect):
    """
    Plots the lanes within the given view rectangle on the map.

    Args:
        plt (matplotlib.pyplot): The matplotlib plotting object.
        vec_map (VectorMap): The vector map containing lane information.
        view_rect (shapely.geometry.box): The view rectangle defining the area to display lanes.

    Returns:
        matplotlib.pyplot: The updated plotting object with lanes plotted.
    """
    for laneid in vec_map.lanes:
        # Access the left and right edges of the lane
        left_lane = laneid.left_edge
        right_lane = laneid.right_edge

        # Plot the left lane edge if it exists and is within the view rectangle
        if left_lane is not None and is_line_in_view(left_lane.points, view_rect):
            plt.plot(left_lane.points[:, 0], left_lane.points[:, 1], '-', markersize=1, color='#969696', linewidth=0.4)

        # Plot the right lane edge if it exists and is within the view rectangle
        if right_lane is not None and is_line_in_view(right_lane.points, view_rect):
            plt.plot(right_lane.points[:, 0], right_lane.points[:, 1], '-', markersize=1, color='#969696', linewidth=0.4)

    return plt


def plot_agent_trajectory(agent_id, interact_ids, all_agents, agent_states, timestamp_index, all_timesteps, dt,
                          vec_map, column_dict, lane_id, lane_kd_tree, sc, agent_lane_ids,
                          considertime):
    """
    Plots the trajectory of a given agent within the scene.

    Args:
        agent_id (str): The ID of the agent whose trajectory is to be plotted.
        interact_ids (list): List of IDs representing interacting agents.
        all_agents (list): List of all agents present in the scene.
        agent_states (np.array): Array of agent states across time steps.
        timestamp_index (int): Index of the current timestamp within the time steps.
        all_timesteps (list): List of all timesteps in the scene.
        dt (float): The time difference between timesteps.
        vec_map (VectorMap): The vector map containing lane information.
        column_dict (dict): Dictionary mapping state columns (e.g., 'x', 'y') to indices.
        lane_id (list): List of lane IDs.
        lane_kd_tree (KDTree): KDTree for spatial lookup of lanes.
        sc (DataFrameCache): Cached data for the scene.
        agent_lane_ids (dict): Dictionary mapping agents to lane IDs over time.
        considertime (float): The time duration considered for trajectory analysis.

    Returns:
        None
    """
    # Get the index of the agent in the list of all agents
    id_index = all_agents.index(agent_id)

    # Set up time range and position indices for processing
    timerange = int(considertime / dt)
    position_index = [column_dict['x'], column_dict['y']]
    v_index = [column_dict['vx'], column_dict['vy']]

    # Process the tracks for the given agent
    processed_tracks = process_tracks_single(
        considertime, agent_states, id_index, timestamp_index,
        timerange, position_index, v_index
    )

    # If no valid tracks are processed, exit the function
    if not processed_tracks:return None

    # Extract processed track information
    line, v = processed_tracks['line'], processed_tracks['velocity']
    line_x, line_y = line.xy
    x, y = agent_states[id_index, timestamp_index, column_dict['x']], agent_states[id_index, timestamp_index, column_dict['y']]
    psi_rad = agent_states[id_index, timestamp_index, column_dict['heading']]

    # Plot based on whether the agent is interacting or non-interacting
    if agent_id in interact_ids:
        plot_interacting_agent(line_x, line_y, x, y, psi_rad)
    else:
        plot_non_interacting_agent(line_x, line_y, x, y, psi_rad)



def plot_interacting_agent(line_x, line_y, x, y, angle):
    """
    Plots the trajectory of an interacting agent, interpolating the trajectory line and drawing the agent's polygon.

    Args:
        line_x (ndarray): The x-coordinates of the trajectory line.
        line_y (ndarray): The y-coordinates of the trajectory line.
        x (float): The x-coordinate of the agent's current position.
        y (float): The y-coordinate of the agent's current position.
        angle (float): The orientation angle of the agent in radians.
    """
    # Interpolate the trajectory line using a linear interpolation function
    f = interp1d(line_x, line_y, kind='linear')
    # Generate new x-values for a smooth line, with 1000 points between min and max of line_x
    xnew = np.linspace(min(line_x), max(line_x), num=1000, endpoint=True)
    ynew = f(xnew)  # Get the corresponding y-values using the interpolation function

    # Plot the interpolated trajectory line
    plt.plot(xnew, ynew, '-', color='#3868A6', zorder=10, linewidth=1)

    # Draw the agent's polygon based on its current position and orientation
    rect = patches.Polygon(polygon_xy_from_motionstate(x, y, angle), closed=True, zorder=20, color='#3868A6')
    plt.gca().add_patch(rect)  # Add the polygon patch to the plot


def plot_non_interacting_agent(line_x, line_y, x, y, angle):
    """
    Plots the trajectory of a non-interacting agent, interpolating the trajectory line and drawing the agent's polygon.

    Args:
        line_x (ndarray): The x-coordinates of the trajectory line.
        line_y (ndarray): The y-coordinates of the trajectory line.
        x (float): The x-coordinate of the agent's current position.
        y (float): The y-coordinate of the agent's current position.
        angle (float): The orientation angle of the agent in radians.
    """
    # Interpolate the trajectory line using a linear interpolation function
    f = interp1d(line_x, line_y, kind='linear')
    # Generate new x-values for a smooth line, with 1000 points between min and max of line_x
    xnew = np.linspace(min(line_x), max(line_x), num=1000, endpoint=True)
    ynew = f(xnew)  # Get the corresponding y-values using the interpolation function

    # Plot the interpolated trajectory line as a dashed line for non-interacting agents
    plt.plot(xnew, ynew, '--', color='#E9E9E9', zorder=5, linewidth=0.8)

    # Draw the agent's polygon based on its current position and orientation
    rect = patches.Polygon(polygon_xy_from_motionstate(x, y, angle), closed=True, zorder=6, color='#C9CACA')
    plt.gca().add_patch(rect)  # Add the polygon patch to the plot



def draw_pic(desired_scene, all_agents, all_timesteps, dataset, id_rawid, raw_scene_id, path_relation,interact_ids, timestamp, start, end, dt, considertime, save_path="./"):
    """
    Draws a scene showing the interaction trajectories of selected agents within a given scene.

    Args:
        desired_scene: The scene object containing information about the scenario.
        all_agents (list): List of all agent IDs present in the scene.
        all_timesteps (range): Range of timesteps within the scene.
        dataset: The dataset object containing information about the scenes.
        id_rawid (dict): Dictionary mapping scene IDs to raw data indices.
        raw_scene_id (int): Raw scene ID to retrieve the desired scene.
        interact_ids (list): List of agent IDs involved in the interaction.
        start (int): Start timestep for the interaction.
        end (int): End timestep for the interaction.
        dt (float): Timestep duration.
        considertime (int): Time range to consider around the interaction.
        save_path (str): Path to save the generated image.

    Returns:
        None
    """
    # Retrieve the vector map and KD-trees for the lanes
    vec_map, lane_kd_tree = get_map_and_kdtrees(dataset, desired_scene)
    lane_id = [lane.id for lane in vec_map.lanes]
    timestamp_index = all_timesteps.index(timestamp)

    # Set up the scene cache and column dictionary
    scene_cache = DataFrameCache(cache_path=dataset.cache_path, scene=desired_scene)
    column_dict = scene_cache.column_dict

    # Get the agent states and lane information
    agent_states, agent_lane_ids = get_agent_states(interact_ids, all_agents, vec_map, lane_kd_tree, scene_cache, desired_scene, column_dict, all_timesteps)

    # Set up plot bounds based on the mean time and interaction details
    mean_time = int((start + end) / 2)
    x_min, x_max, y_min, y_max = setup_plot_bounds(
        interact_ids, mean_time, scene_cache, column_dict, vec_map, all_agents,
        all_timesteps, dt, considertime
    )

    # Enable interactive plotting
    plt.ion()

    # Plot lanes within the view rectangle
    plot_lanes(plt, vec_map, box(x_min, y_min, x_max, y_max))

    # Get current agents present in the scene at the given timestamp
    current_ids = [
        all_agents[index] 
        for index, flag in enumerate(agent_states[:, timestamp_index, column_dict['vx']]) 
        if flag != 0
    ]

    # Plot each agent's trajectory
    for agent_id in current_ids:
        plot_agent_trajectory(
            agent_id, interact_ids, all_agents, agent_states, timestamp_index, 
            all_timesteps, dt, vec_map, column_dict, lane_id, 
            lane_kd_tree, scene_cache, agent_lane_ids, considertime
        )

    # Set up the plot appearance
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Track {interact_ids} at Timestamp {timestamp}_{path_relation}')
    plt.xlim(x_min, x_max)  # Set X axis range
    plt.ylim(y_min, y_max)  # Set Y axis range

    plt.gca().set_facecolor('xkcd:white')
    plt.gca().margins(0)
    plt.gca().set_aspect('equal')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.tight_layout()

    # Create directory if it doesn't exist and save the plot
    os.makedirs(save_path, exist_ok=True)
    interact_ids_str = '_'.join(interact_ids)
    plt.savefig(f'{save_path}/{interact_ids_str}_{timestamp}.png', dpi=600)

    #  Clear the plot
    plt.clf()

