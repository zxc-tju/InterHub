import numpy as np
from shapely.geometry import LineString, Point


def calculate_angle(p1, p2):
    """Calculate the angle between two points (p1 and p2) in degrees"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.arctan2(dy, dx)  # Angle in radians
    angle_deg = np.degrees(angle)  # Convert radians to degrees
    return angle_deg



def get_safe_trajectory_slice(coords, start_idx=-51, end_idx=-1, min_dist=0.1, min_points=10):
    """Safe trajectory slicing with validation"""
    coords = np.asarray(coords)
    n = len(coords)
    
    # Handle negative indices
    start = max(0, n + start_idx if start_idx < 0 else start_idx)
    end = min(n, n + end_idx if end_idx < 0 else end_idx)
    
    # Get initial slice and filter points
    slice = coords[start:end]
    filtered = [slice[0]]
    last_point = slice[0]
    
    for point in slice[1:]:
        if np.linalg.norm(point - last_point) > min_dist:
            filtered.append(point)
            last_point = point
    
    # If not enough points, try extending window
    while len(filtered) < min_points and start > 0:
        start = max(0, start - 10)
        slice = coords[start:end]
        filtered = [slice[0]]
        last_point = slice[0]
        for point in slice[1:]:
            if np.linalg.norm(point - last_point) > min_dist:
                filtered.append(point)
                last_point = point
    
    return np.array(filtered) if len(filtered) >= 2 else slice


class TrajectoryProcessor:
    
    def _get_closest_index(self, trajectory, point):
        """Find the index in the trajectory closest to a given point"""
        if isinstance(trajectory, LineString):
            trajectory = np.array(trajectory.coords)
        point = np.array([point.x, point.y])
        distances = np.linalg.norm(trajectory - point, axis=1)
        return np.argmin(distances)

    def _get_trajectory_window(self, trajectory, point, time_window, is_after=True):
        """Generic function to extract a trajectory window before or after a given point"""

        # Get coordinates and closest index
        coords = np.array(trajectory.coords) if isinstance(trajectory, LineString) else np.array(trajectory)
        point = np.array([point.x, point.y])
        closest_idx = np.argmin(np.linalg.norm(coords - point, axis=1))
        
        # Calculate window bounds
        if is_after:
            start = max(0, closest_idx - 2)  # Include 2 points before for context
            end = min(closest_idx + time_window + 1, len(coords))
        else:
            start = max(0, closest_idx - time_window - 1)
            end = min(closest_idx + 2, len(coords))  # Include 2 points after for context
        
        # Return as LineString
        return LineString(coords[start:end])
        
    

    def get_trajectory_after_point(self, trajectory, point, time_window):
        """Get trajectory segment after a given point"""
        return LineString(self._get_trajectory_window(trajectory, point, time_window, is_after=True))

    def get_trajectory_before_point(self, trajectory, point, time_window):
        """Get trajectory segment before a given point"""
        return LineString(self._get_trajectory_window(trajectory, point, time_window, is_after=False))

    def get_effective_line(self, line, start_idx, end_idx):
        """Get a cleaned slice of the trajectory"""
        return get_safe_trajectory_slice(np.array(line.coords), start_idx, end_idx)

    def calculate_angle(self, line, default_start, default_end):
        """Calculate the trajectory direction angle based on a line segment or fallback points"""
        return calculate_angle(line[0], line[-1]) if len(line) >= 2 else calculate_angle(default_start, default_end)
    
    def angle_difference(self, angle1, angle2):
        """Calculate the difference between two angles in degrees, ensuring it is between -180 and 180"""
        diff = angle2 - angle1
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def determine_intersect(self, line1, line2):
        """Check if two trajectory lines intersect (including offset variants)"""
        return any([
            line1.intersects(line2.parallel_offset(1, side))
            for side in ('left', 'right')
            for line in [line1, line2]
        ])

    def get_crossing_type(self, angle_diff, threshold, is_after=False):
        """Determine crossing type based on angle difference"""
        if -threshold <= angle_diff <= threshold:
            return 'P'
        elif (-(180-threshold) < angle_diff < -threshold) or (threshold < angle_diff < 180-threshold):
            return 'C'
        elif is_after and abs(angle_diff) >= 180-threshold:
            return 'O'
        elif not is_after:
            return 'O'
        else:
            return 'UNKNOWN'





class Segment_Labeler:
    def __init__(self, agent_states, all_agents, key_pair, end_time, intersection_point, fut_line1, fut_line2, pair):
        self.agent_states = agent_states
        self.all_agents = all_agents
        self.key_pair = key_pair
        self.end_time = end_time
        self.intersection = intersection_point
        self.fut_line1, self.fut_line2 = fut_line1, fut_line2
        self.interaction_ids = pair

    def _get_lines(self):
        def get_motion_state(agent_states, all_agents, key_agent):
            """
            Get trajectory for specified agent IDs
            """
            agent_id_to_index = {agent_id: idx for idx, agent_id in enumerate(all_agents)}
            indices_to_query = [agent_id_to_index[agent_id] for agent_id in key_agent]
            motion_state = agent_states[indices_to_query, :, :2]

            self.indices_to_query = indices_to_query
            return motion_state
        
        def filter_zero_points(motion_state):
            """
            Filter out trajectory points containing zeros
            """
            filtered_track = motion_state[np.any(motion_state != 0, axis=1)]
            return LineString(filtered_track) if len(filtered_track) >= 2 else None

        def concat_lines(motion_line, fut_line):
            """Concatenate two LineStrings if both exist, otherwise return the motion_line"""
            return LineString(list(motion_line.coords) + list(fut_line.coords)) \
                if (motion_line and fut_line) else motion_line or fut_line

        motion_state = get_motion_state(self.agent_states, self.all_agents, self.key_pair)

        self.agent1_line = concat_lines(filter_zero_points(motion_state[0]), self.fut_line1)
        self.agent2_line = concat_lines(filter_zero_points(motion_state[1]), self.fut_line2)

    def _get_agent_state(self, turns_angle, s_t_angle, o_t_angle):
        """Determine agent turn type"""
        if s_t_angle < turns_angle < o_t_angle:
            agent_state = 'R'
        elif -o_t_angle < turns_angle < -s_t_angle:
            agent_state = 'L'
        elif abs(turns_angle) < s_t_angle:
            agent_state = 'S'
        else:
            agent_state = 'U'

        return agent_state

    def get_path_relation_label(self, after_window=250, before_window=150, threshold=30):
        """Generate the secondary path interaction label"""
        trajectoryprocessor = TrajectoryProcessor()

        self._get_lines()

        # Get trajectory segments before and after the interaction
        after_traj = {
            'a1': trajectoryprocessor.get_trajectory_after_point(self.agent1_line, self.intersection, after_window),
            'a2': trajectoryprocessor.get_trajectory_after_point(self.agent2_line, self.intersection, after_window)
        }
        before_traj = {
            'a1': trajectoryprocessor.get_trajectory_before_point(self.agent1_line, self.intersection, before_window),
            'a2': trajectoryprocessor.get_trajectory_before_point(self.agent2_line, self.intersection, before_window)
        }

        # Extract effective slices from the trajectories
        after_effective = {
            'a1': trajectoryprocessor.get_effective_line(after_traj['a1'], -51, -1),
            'a2': trajectoryprocessor.get_effective_line(after_traj['a2'], -51, -1)
        }
        before_effective = {
            'a1': trajectoryprocessor.get_effective_line(before_traj['a1'], 0, 50),
            'a2': trajectoryprocessor.get_effective_line(before_traj['a2'], 0, 50)
        }

        pre_int_i, post_int_i, pre_int_j, post_int_j = len(before_effective['a1']), len(after_effective['a1']), len(before_effective['a2']), len(after_effective['a2'])

        # Compute entry and exit angles
        after_angles = {
            'a1': trajectoryprocessor.calculate_angle(after_effective['a1'], self.intersection.coords[0], np.array(after_traj['a1'].coords)[-1]),
            'a2': trajectoryprocessor.calculate_angle(after_effective['a2'], self.intersection.coords[0], np.array(after_traj['a2'].coords)[-1])
        }
        before_angles = {
            'a1': trajectoryprocessor.calculate_angle(before_effective['a1'], np.array(before_traj['a1'].coords)[0], self.intersection.coords[0]),
            'a2': trajectoryprocessor.calculate_angle(before_effective['a2'], np.array(before_traj['a2'].coords)[0], self.intersection.coords[0])
        }
        angle_diff_after = trajectoryprocessor.angle_difference(after_angles['a1'], after_angles['a2'])
        angle_diff_before = trajectoryprocessor.angle_difference(before_angles['a1'], before_angles['a2'])

        # Check whether the trajectories intersect
        before_intersects = trajectoryprocessor.determine_intersect(before_traj['a1'], before_traj['a2'])
        after_intersects = trajectoryprocessor.determine_intersect(after_traj['a1'], after_traj['a2'])

        # Determine crossing types
        before_type = 'M' if ((not before_intersects) & (-threshold <= angle_diff_before <= threshold)) else trajectoryprocessor.get_crossing_type(angle_diff_before, threshold)
        after_type = 'M' if ((not after_intersects) & (-threshold <= angle_diff_after <= threshold)) else trajectoryprocessor.get_crossing_type(angle_diff_after, threshold, is_after=True)

        # Determine the path relation label
        # Combine the final path relation label
        path_rel = f"{before_type}-{after_type}"

        path_relation = 'F' if before_type == 'M' else path_rel

        self.after_angles, self.before_angles = after_angles, before_angles
        self.after_effective, self.before_effective = after_effective, before_effective
        self.path_relation = path_relation
        return path_relation, pre_int_i, post_int_i, pre_int_j, post_int_j
    

    def get_path_category(self, path_relation):
        """Classify path relation label into a higher-level path category"""
        MP_list = ['P-M', 'C-M', 'O-M', 'P-P', 'C-P', 'O-P']
        CP_list = ['P-C', 'C-C', 'O-C', 'C-O']
        HO_list = ['P-O', 'O-O']
        F_list = ['F']

        if path_relation in MP_list:
            path_category = 'MP'
        elif path_relation in CP_list:
            path_category = 'CP'
        elif path_relation in HO_list:
            path_category = 'HO'
        elif path_relation in F_list:
            path_category = 'F'
        return path_category


    def get_turn_label(self, s_t_angle=10, o_t_angle=170):
        """To be completed: compute turn label based on before/after angle change"""
        trajectoryprocessor = TrajectoryProcessor()
        turn_angle1 = trajectoryprocessor.angle_difference(self.after_angles['a1'], self.before_angles['a1'])
        turn_angle2 = trajectoryprocessor.angle_difference(self.after_angles['a2'], self.before_angles['a2'])
        
        agent_turn_state1 = self._get_agent_state(turn_angle1, s_t_angle, o_t_angle)
        agent_turn_state2 = self._get_agent_state(turn_angle2, s_t_angle, o_t_angle)

        self.agent_turn_state1 = agent_turn_state1
        self.agent_turn_state2 = agent_turn_state2
        return agent_turn_state1 + '-' + agent_turn_state2
    
    def get_agent_lanes(self, agent_coords, all_lanes):
        """
        Get the lanes corresponding to the agent at each timestep.
        Args:
            agent_coords (np.ndarray): Array of agent coordinates including position and heading.
            all_lanes (dict): Dictionary of all lanes, where key is lane_id and value is the lane object.
        Returns:
            dict: Dictionary with (x, y, heading) as keys and corresponding lane objects as values.
        """
        agent_all_lanes = {}
        for i in range(len(agent_coords)):
            if agent_coords[i, 7] != 0.0:  # Check if lane_id is valid
                coords_key = tuple(agent_coords[i, [0, 1, 7]])  # (x, y, heading)
                lane_id = int(agent_coords[i, -1])
                agent_all_lanes[coords_key] = all_lanes[str(lane_id)]
        return agent_all_lanes

    def _get_nearest_lane_heading(self, point, lane):
        """"Get the heading of the nearest point on the lane center line to the input point."""
        center_line = LineString(lane.center.points[:, :2])
        nearest_point = center_line.interpolate(center_line.project(Point(point)))
        distances = [nearest_point.distance(Point(p)) for p in center_line.coords]
        return lane.center.points[np.argmin(distances), 3]

    def get_lane_in_heading(self, agent_all_lanes):
        """Select lanes that match the agent's heading direction."""
        lane_in_heading = []
        for (x, y, vehicle_heading), lane in agent_all_lanes.items():
            lane_heading = self._get_nearest_lane_heading((x, y), lane)
            if abs(vehicle_heading - lane_heading) < np.pi/8:
                lane_in_heading.append(lane)
        return lane_in_heading

    def _build_lane_relation(self, lanes):
        """Build a lane connectivity graph including next, previous, and adjacent lanes."""
        lanes_id = [lane.id for lane in lanes]
        relation = {
            lane.id: lane.next_lanes | lane.prev_lanes | 
                    lane.adj_lanes_left | lane.adj_lanes_right 
            for lane in lanes
        }
        return {
            lane_id: {
                conn_id for conn_id in connected_lanes 
                if conn_id in relation.keys() and 
                lanes_id.index(conn_id) > lanes_id.index(lane_id)
            }
            for lane_id, connected_lanes in relation.items()
        }


    def find_lane_path(self, relation, start_id, end_id):
        """
        Find a path from start_id to end_id using the given lane connectivity.
        """
        def dfs(current_id, visited, path):
            if current_id == end_id:
                return True

            visited.add(current_id)
            for neighbor_id in relation.get(current_id, []):
                if neighbor_id not in visited:
                    path.append(neighbor_id)
                    current_id = neighbor_id

                    if dfs(neighbor_id, visited, path):
                        return True

                    path.pop()

            return False
    
    def get_continuous_lanes(self, lanes):
        """
        Get a sequence of connected lanes.
        Args:
            lanes (list): List of lane objects.
        Returns:
            list: A list of connected lane objects.
        """
        if not lanes:
            return []
            
        relation = self._build_lane_relation(lanes)
        if not any(relation.values()):
            return [lanes[0]]
            
        lanes_id = [lane.id for lane in lanes]
        start_id = next((k for k, v in relation.items() if v), lanes_id[0])
        
        for end_idx in range(len(lanes_id)-1, lanes_id.index(start_id), -1):
            path = self.find_lane_path(relation, start_id, lanes_id[end_idx])
            if path:
                return [lane for lane in lanes if lane.id in path]
                
        for start_idx in range(lanes_id.index(start_id)+1, len(lanes_id)-1):
            for end_idx in range(len(lanes_id)-1, start_idx, -1):
                path = self.find_lane_path(relation, lanes_id[start_idx], lanes_id[end_idx])
                if path:
                    return [lane for lane in lanes if lane.id in path]
        return []

    def get_lane_boundaries(self, lanes, boundary_type):
        """Get left or right boundaries of the given lanes."""
        return [
            LineString(getattr(lane, f"{boundary_type}_edge").points[:, :2])
            for lane in lanes if getattr(lane, f"{boundary_type}_edge")
        ]

    def _check_boundary_crossing(self, agent_line, boundaries):
        """Check if the agent trajectory line crosses any lane boundaries."""
        intersection_points = []
        for boundary in boundaries:
            if agent_line.intersects(boundary):
                intersection = agent_line.intersection(boundary)
                if intersection.geom_type == 'Point':
                    intersection_points.append((intersection.x, intersection.y))
                elif intersection.geom_type == 'MultiPoint':
                    intersection_points.extend((p.x, p.y) for p in intersection.geoms)
        return bool(intersection_points), intersection_points

    def change_lane_priority(self, all_lanes):
        """Main function to determine priority based on lane boundary crossings."""
        agent1_coords, agent2_coords = self.agent_states[self.indices_to_query, :, :]
        
        agent1_lanes = self.get_lane_in_heading(self.get_agent_lanes(agent1_coords, all_lanes))
        agent2_lanes = self.get_lane_in_heading(self.get_agent_lanes(agent2_coords, all_lanes))
        
        agent1_cont_lanes = self.get_continuous_lanes(agent1_lanes)
        agent2_cont_lanes = self.get_continuous_lanes(agent2_lanes)
        
        agent1_bounds = self.get_lane_boundaries(agent1_cont_lanes, 'left') + \
                       self.get_lane_boundaries(agent1_cont_lanes, 'right')
        agent2_bounds = self.get_lane_boundaries(agent2_cont_lanes, 'left') + \
                       self.get_lane_boundaries(agent2_cont_lanes, 'right')
        
        agent1_line = LineString(np.vstack((self.before_effective['a1'], self.after_effective['a1'][:-10])))
        agent2_line = LineString(np.vstack((self.before_effective['a2'], self.after_effective['a2'][:-10])))
        
        agent1_crosses, _ = self._check_boundary_crossing(agent1_line, agent2_bounds)
        agent2_crosses, _ = self._check_boundary_crossing(agent2_line, agent1_bounds)
        
        if agent1_crosses and not agent2_crosses:
            return (1, 0)  
        elif not agent1_crosses and agent2_crosses:
            return (0, 1)  
        return 'Unknown' 


    def get_priority_label(self, all_lanes):
        """Determine interaction priority label based on turning direction and lane crossing."""
        priority_order = ['S', 'L', 'R', 'U']
        s1, s2 = self.agent_turn_state1, self.agent_turn_state2

        # Determine priority based on turn states
        if s1 != s2:
            priority_tuple = (0, 1) if priority_order.index(s1) < priority_order.index(s2) else (1, 0)
        elif s1 == 'S' and self.path_relation in {'P-M', 'P-P'}:
            priority_tuple = self.change_lane_priority(all_lanes)
        else:
            priority_tuple = (0, 0)

        # Handle special case for path relation 'F'
        if self.path_relation == 'F':
            return 'Unknown'

        # Map priority tuple to labels
        return {
            'Unknown': 'Unknown',
            (0, 1): self.interaction_ids[0],
            (1, 0): self.interaction_ids[1],
            (0, 0): 'equal'
        }.get(priority_tuple, 'Unknown')