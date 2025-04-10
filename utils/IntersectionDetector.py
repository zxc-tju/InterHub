from itertools import combinations
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points

class IntersectionDetector:
    """Class to detect intersections between vehicle trajectories within a future time window."""

    def __init__(self, agent_states, all_timesteps, column_dict, 
                 all_agents, move_agent, map_processed_data):
        """Initialize the IntersectionDetector class with required attributes.

        Args:
            agent_states: The states of all agents.
            all_timesteps: List of all time steps.
            column_dict: Dictionary mapping column names to indices.
            all_agents: Dictionary of all agents in the scene.
            move_agent: Dictionary of moving agents per timestamp.
        """
        # Initialize class attributes
        (self.agent_states,
         self.all_timesteps, self.column_dict,
         self.all_agents, self.move_agent,
         self.map_processed_data) = (agent_states,
                                                   all_timesteps, column_dict, all_agents, 
                                                   move_agent, map_processed_data)

    def intersection_detector(self):
        """Detect intersections for all vehicle pairs within the considered time range.

        Returns:
            dict: A dictionary containing information about vehicle pairs and intersections at each timestamp.
        """
        time_dict = {}

        # Iterate over each timestamp
        for timestamp in self.all_timesteps:
            current_ids = self.move_agent[timestamp]
            pair_list = list(combinations(current_ids, 2))

            # Calculate collision points for all vehicle pairs
            point_list, dis_list, track_ids, collision_point_list = self.find_collisionpoint(timestamp, pair_list)

            # Create dictionaries to store distance and collision point information
            pair_dis_dic = {(pair[0], pair[1]): dis for pair, dis in zip(track_ids, dis_list) if dis[2]}
            pair_point_dic = {(pair[0], pair[1]): point for pair, point in zip(track_ids, point_list) if point[0]}
            intersection_dic = {(pair[0], pair[1]): point for pair, point in zip(track_ids, collision_point_list)}

            # Store information in time_dict
            time_dict[timestamp] = {
                'id_list': current_ids,
                'pair_list': pair_dis_dic.keys(),
                'pair_dis_dic': pair_dis_dic,
                'pair_point_dic': pair_point_dic,
                'intersection_dic': intersection_dic
            }

        return time_dict

    def find_collisionpoint(self, timestamp, pair_list):
        """Find collision points between all vehicle pairs at a specific timestamp.

        Args:
            timestamp: The current timestamp to check for collisions.
            pair_list: List of vehicle pairs to check for intersections.

        Returns:
            Tuple containing point list, distance list, track IDs, and collision points.
        """
        index_time = self.all_timesteps.index(timestamp)
        position_index = [self.column_dict['x'], self.column_dict['y']]

        point_list, dis_list, track_ids, collision_point_list = [], [], [], []

        for track_id1, track_id2 in pair_list:
            all_ids = list(self.all_agents.keys())
            track_id_index1, track_id_index2 = all_ids.index(track_id1), all_ids.index(track_id2)

            line1 = self.map_processed_data[timestamp].get(track_id1, {}).get('line', None)
            line2 = self.map_processed_data[timestamp].get(track_id2, {}).get('line', None)
            line2_0 = line2
            v1 = self.map_processed_data[timestamp].get(track_id1, {}).get('velocity', 1e-6)  # Avoid division by zero
            v2 = self.map_processed_data[timestamp].get(track_id2, {}).get('velocity', 1e-6)

            if line1 is None or line2 is None:
                continue

            intersection = line1.intersection(line2)
            if intersection.is_empty:
                continue

            x1, y1 = self.agent_states[track_id_index1, index_time, position_index]
            _, collision_point = nearest_points(Point(x1, y1), intersection)
            collision_point_x, collision_point_y = collision_point.x, collision_point.y

            dis1, dis2 = line1.project(collision_point), line2.project(collision_point)
            t_diff = abs(dis1 / v1 - dis2 / v2)
            valid = True

            # Determine front and rear vehicle positions
            if dis1 > dis2:
                track_id1, track_id2 = track_id2, track_id1
                dis1, dis2 = dis2, dis1
                line1, line2 = line2, line1
                track = self.agent_states[track_id_index2, index_time - 4:index_time, position_index]
            else:
                track = self.agent_states[track_id_index1, index_time - 4:index_time, position_index]

            filtered_track = track[~np.all(track == 0, axis=1)]
            if filtered_track.shape[0] >= 2:
                hist = LineString(filtered_track.T)
                combined_points = list(hist.coords) + list(line1.coords)
                line1 = LineString(combined_points)

            dis_buffer = dis2
            left_buffer = line1.parallel_offset(1.5, 'left')
            right_buffer = line1.parallel_offset(1.5, 'right')
            combined_coords = list(left_buffer.coords) + list(right_buffer.coords)[::-1]

            if len(combined_coords) < 4:
                continue

            combined_polygon = Polygon(combined_coords)
            intersects = line2.intersects(left_buffer) or line2.intersects(right_buffer)
            if intersects:
                dis_buffer = line2.project(self.buffer_point(line2, left_buffer, right_buffer))
            contains_point = combined_polygon.contains(line2.interpolate(dis_buffer - 1))

            if not intersects or (intersects and contains_point):
                valid = False

            point_list.append((collision_point_x, collision_point_y))
            dis_list.append((dis1, dis2, valid, t_diff))
            track_ids.append((track_id1, track_id2))
            collision_point_list.append(collision_point)

            import matplotlib.pyplot as plt
            # if valid:
            #     # Plotting code
            #     plt.figure(figsize=(8, 6))
            #     plt.plot(*line1.xy, label=f'Line1 (Track {track_id1})', color='blue')
            #     plt.plot(*line2_0.xy, label=f'Line2_0 (Track {track_id2})', color='green')
            #     plt.plot(*line2.xy, label=f'Line2 (Track {track_id2})', color='orange')
            #
            #     # Plot the collision point
            #     plt.scatter(collision_point_x, collision_point_y, color='red', label='Collision Point', zorder=5)
            #
            #     # Plot buffers
            #     plt.plot(*left_buffer.xy, linestyle='--', color='black', label='Left Buffer')
            #     plt.plot(*right_buffer.xy, linestyle='--', color='black', label='Right Buffer')
            #
            #     # Plot combined polygon
            #     x, y = combined_polygon.exterior.xy
            #     plt.fill(x, y, alpha=0.3, color='gray', label='Combined Polygon')
            #
            #     plt.xlabel('X-coordinate')
            #     plt.ylabel('Y-coordinate')
            #     plt.title(f'Collision Visualization for Track {track_id1} and Track {track_id2}')
            #     plt.legend()
            #     plt.grid(True)
            #     plt.show()

        return point_list, dis_list, track_ids, collision_point_list

    def buffer_point(self, line, left_buffer, right_buffer):
        """Find the closest intersection point between the line and its buffers.

        Args:
            line: The line representing the vehicle's trajectory.
            left_buffer: The left buffer of the line.
            right_buffer: The right buffer of the line.

        Returns:
            Point: The closest intersection point.
        """
        intersection_left = line.intersection(left_buffer)
        intersection_right = line.intersection(right_buffer)
        line_start = Point(line.xy[0][0], line.xy[0][1])
        intersections = []

        if not intersection_left.is_empty:
            if isinstance(intersection_left, Point):
                intersections.append(intersection_left)
            else:
                intersections.extend(intersection_left.geoms)

        if not intersection_right.is_empty:
            if isinstance(intersection_right, Point):
                intersections.append(intersection_right)
            else:
                intersections.extend(intersection_right.geoms)

        if intersections:
            closest_intersection = min(intersections, key=lambda p: line_start.distance(p))
            return closest_intersection
