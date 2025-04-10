import numpy as np
import math
from scipy.optimize import minimize

class MultiProcess():
    def find_related_groups(self, valid_keys):
        """
        Find groups of related elements based on the tuples in valid_keys.
        
        :param valid_keys: List of tuples where each tuple represents a connection.
        :return: A list of groups, where each group is a tuple of related elements.
        """
        # Dictionary to store the relationships between elements in the tuples
        tuple_dict = {}

        # Initialize the dictionary to associate each element with its related elements
        for tup in valid_keys:
            for element in tup:
                tuple_dict.setdefault(element, set()).update(tup)

        # Build connected groups
        visited = set()
        groups = []

        for tup in valid_keys:
            group = set()
            stack = list(tup)

            # Traverse each element in the stack to build the group
            while stack:
                element = stack.pop()
                if element not in visited:
                    visited.add(element)
                    group.add(element)
                    stack.extend(tuple_dict[element] - visited)

            # Add the group to the list if it's not empty
            if group:
                groups.append(tuple(group))

        return groups

    def extract_interactions(self, time_dic):
        """
        Extract interaction information based on the provided time dictionary.
        
        :param time_dic: Dictionary where keys are time steps and values are dictionaries containing 'pair_dis_dic'.
        :return: A dictionary with time steps as keys and interaction details as values.
        """
        result_dict = {}

        for time_step, dic in time_dic.items():
            result_dict[time_step] = {}
            interaction_info = {}

            # Extract the pair distance dictionary
            pair_dis_dic = dic['pair_dis_dic']
            valid_keys = [pair for pair, distances in pair_dis_dic.items() if distances[2]]

            # Find related unique key groups
            unique_tuples = self.find_related_groups(valid_keys)
   
            # Iterate through each unique group
            for pair in unique_tuples:
                interaction_info[pair] = {}
                interaction_info[pair] = {key: pair_dis_dic[key] for key in valid_keys if any(i in pair for i in key)}
                
                # Merge related elements into a single tuple
                merged_tuple = tuple(sorted(set(element for sub_pair in interaction_info[pair].keys() for element in sub_pair)))
                
                # Update the result dictionary with the merged tuple and corresponding data
                result_dict[time_step][merged_tuple] = {sub_pair: sub_distances for sub_pair, sub_distances in interaction_info[pair].items()}

        return result_dict

class AccelerationCalculate(MultiProcess):

    def constraint_function1(self, a, d, v, ai, ti):
        t1, t2 = a[ti[0]], a[ti[1]]
        delta_TTCP = abs(t1 - t2)
        return delta_TTCP - 1.5

    def constraint_function2(self, a, d, v, ai, ti):
        d1 = d[0]
        v1 = v[0]
        t1 = a[ti[0]]
        dis1 = v1 * t1 + a[0] * t1 ** 2 / 2
        return 0.1 - abs(dis1 - d1)

    def constraint_function3(self, a, d, v, ai, ti):
        d2 = d[1]
        v2 = v[1]
        t2 = a[ti[1]]
        dis2 = v2 * t2 + a[1] * t2 ** 2 / 2
        return 0.1 - abs(d2 - dis2)

    def constraint_function4(self, a, d, v, ai, ti):
        v1 = v[0]
        t1 = a[ti[0]]
        dis1 = v1 * t1 + a[0] * t1 ** 2 / 2
        return dis1

    def constraint_function5(self, a, d, v, ai, ti):
        v2 = v[1]
        t2 = a[ti[1]]
        dis2 = v2 * t2 + a[1] * t2 ** 2 / 2
        return dis2
    
    def constraint_function6(self, a, d, v, ai, ti):
        v1 = v[0]
        t1 = a[ti[0]]
        v1 = v1 + a[0] * t1 
        return v1
    
    def constraint_function7(self, a, d, v, ai, ti):
        v2 = v[1]
        t2 = a[ti[1]]
        v2 = v2 + a[1] * t2 
        return v2


    def MultiCalculateAcceleration(self, pair_info, pair_multi, timestamp_now):
        pet_list = []
        num = len(pair_multi)
        
        # Objective function to minimize the sum of absolute accelerations
        def objective_function(a):
            return sum(abs(a[i]) for i in range(num))

        # Initial guess for acceleration values (one for each vehicle)
        initial_guess = [0] * num
        bounds = [(None, None)] * num  # No specific bounds on acceleration values
        total_constraints = []
        
        # Get the keys from pair_info, representing vehicle pairs
        pair_info_keys = list(pair_info.keys())

        # Iterate through each pair and set up constraints and initial guesses
        for pair, info in pair_info.items():
            # Get the index for each vehicle in the pair
            ai_1, ai_2 = pair_multi.index(pair[0]), pair_multi.index(pair[1])
            # Calculate time indices based on the pair's position in pair_info_keys
            ti_1 = pair_info_keys.index(pair) * 2 + num
            ti_2 = ti_1 + 1
            # Retrieve distance and velocity information for the vehicles
            d1, d2 = info['dis']
            v1, v2 = info['motion1'][-1], info['motion2'][-1]

            # Ensure the vehicles are correctly ordered based on their distances
            if d1 >= d2:
                d, v, ai, ti = [d2, d1], [v2, v1], [ai_2, ai_1], [ti_2, ti_1]
            else:
                d, v, ai, ti = [d1, d2], [v1, v2], [ai_1, ai_2], [ti_1, ti_2]

            # Set initial values and bounds based on velocity conditions
            if v1 == 0 or v2 == 0:
                initial_guess += [999, 999]  # Large values as initial guesses for time
                bounds += [(0, None), (0, None)]
            else:
                initial_guess += [d1 / v1, d2 / v2]  # Calculate initial time guesses based on distances and velocities
                bounds += [(0, None), (0, None)]

            # Define the constraints for this vehicle pair
            constraints = [
                {'type': 'ineq', 'fun': self.constraint_function1, 'args': (d, v, ai, ti)},
                {'type': 'ineq', 'fun': self.constraint_function2, 'args': (d, v, ai, ti)},
                {'type': 'ineq', 'fun': self.constraint_function3, 'args': (d, v, ai, ti)},
                {'type': 'ineq', 'fun': self.constraint_function4, 'args': (d, v, ai, ti)},
                {'type': 'ineq', 'fun': self.constraint_function5, 'args': (d, v, ai, ti)},
                {'type': 'ineq', 'fun': self.constraint_function6, 'args': (d, v, ai, ti)},
                {'type': 'ineq', 'fun': self.constraint_function7, 'args': (d, v, ai, ti)}
            ]
            total_constraints.extend(constraints)
            # Calculate PET (Post Encroachment Time) and add to list
            pet_list.append(self.dif_TTCP(d1, d2, v1, v2))

        # Minimize the objective function with constraints and bounds
        result = minimize(objective_function, initial_guess, constraints=total_constraints, bounds=bounds)
        # If the optimization fails, retry with increased max iterations
        if not result.success:
            for maxiter in [1000, 2000]:
                result = minimize(objective_function, result.x, constraints=total_constraints, bounds=bounds,
                                  options={'maxiter': maxiter}, tol=1e-3)
                if result.success:
                    break
            # If still not successful, return default values
            if not result.success:
                print('Optimization failed:', result.message)
                return 0, True, [0] * len(pair_multi), min(pet_list), np.mean(pet_list)

        # Return the results including the minimum PET and the mean PET
        return result.fun, result.success, result.x, min(pet_list), np.mean(pet_list)


    def dif_TTCP(self, d1, d2, v1, v2):
        """
        Calculate the difference in Time-to-Collision Point (TTCP) between two vehicles.

        Parameters:
        d1 (float): Distance of the first vehicle to the collision point.
        d2 (float): Distance of the second vehicle to the collision point.
        v1 (float): Speed of the first vehicle.
        v2 (float): Speed of the second vehicle.

        Returns:
        float: The absolute difference in TTCP between the two vehicles. 
            If either vehicle speed is close to zero, returns a large default value (100) as a fallback.
        """
        
        # Check if both vehicles have significant speed (greater than 0.001)
        if v1 > 0.001 and v2 > 0.001:
            TTCP_1 = d1 / v1  # Calculate TTCP for the first vehicle
            TTCP_2 = d2 / v2  # Calculate TTCP for the second vehicle
            delta_TTCP = abs(TTCP_1 - TTCP_2)  # Calculate the absolute difference between the two TTCP values
            return delta_TTCP
        else:
            # If either vehicle speed is very low, return a large default value
            delta_TTCP = 100
            return delta_TTCP


    def extract_motion(self, agent_states, agent_id, timestamp, index_x, index_y, index_vx, index_vy):
        """
        Extracts the motion information of a given agent at a specific timestamp.

        Parameters:
            agent_states (array-like): The state information of all agents over time.
            agent_id (str): The ID of the agent whose motion information is to be extracted.
            timestamp (int): The specific timestamp to extract the motion data.
            index_x (int): The index for the x-coordinate in the agent state array.
            index_y (int): The index for the y-coordinate in the agent state array.
            index_vx (int): The index for the velocity in the x-direction in the agent state array.
            index_vy (int): The index for the velocity in the y-direction in the agent state array.

        Returns:
            tuple: A tuple containing the x and y coordinates, velocities in the x and y directions, and the calculated speed.
        """

        # Get a list of all agent IDs in the dataset
        all_ids = list(self.all_agents.keys())
        
        # Retrieve the motion information for the specified agent at the given timestamp
        current_motion = agent_states[all_ids.index(agent_id)][timestamp]
        
        # Extract the x, y coordinates and velocities in the x, y directions
        motion = (current_motion[index_x], current_motion[index_y], 
                current_motion[index_vx], current_motion[index_vy])
        
        # Calculate the speed as the magnitude of the velocity vector
        speed = math.sqrt(motion[2] ** 2 + motion[3] ** 2)
        
        # Return the motion information along with the speed
        return motion + (speed,)


    def get_acceleration(self, time_dict_scenario, agent_states, all_agents, column_dict, all_timesteps):
        """
        Calculates the acceleration for each pair of interacting agents over time.

        Parameters:
            time_dict_scenario (dict): A dictionary containing interaction information for each timestamp.
            agent_states (array-like): The state information of all agents over time.
            all_agents (dict): A dictionary of all agents and their IDs.
            column_dict (dict): A dictionary mapping state variables (e.g., 'x', 'vx') to their indices.
            all_timesteps (list): A list of all timestamps to be considered.

        Returns:
            dict: A dictionary with timestamps as keys and dictionaries containing acceleration information as values.
        """

        # Set instance variables for agent states, agents, column dictionary, and timesteps
        self.agent_states = agent_states
        self.all_agents = all_agents
        self.column_dict = column_dict
        self.all_timesteps = all_timesteps

        # Extract the indices for x, y coordinates and velocities in x, y directions
        index_x, index_y = self.column_dict['x'], self.column_dict['y']
        index_vx, index_vy = self.column_dict['vx'], self.column_dict['vy']

        # Dictionary to store acceleration information for each timestamp
        timestamp_a = {}
        sub_pair_value = {}
        # Iterate through each timestamp
        for timestamp_now in self.all_timesteps:
            current_a_min_dict = {}
            
            # Skip the timestamp if there's no data for it in the scenario dictionary
            if timestamp_now not in time_dict_scenario:
                continue

            # Get interaction data for the current timestamp
            current_time_data = time_dict_scenario[timestamp_now]
            
            # Iterate over each pair of interacting agents (multiple agents per pair)
            for pair_multi, pair_dis in current_time_data.items():  # pair_multi represents the group of interacting agents
                pair_info = {}

                # Iterate through each pair within the multi-agent interaction group
                for (id_1, id_2), (dis1, dis2, _, t_diff) in pair_dis.items():
                    # Extract motion data for each agent in the pair
                    motion1 = self.extract_motion(self.agent_states, id_1, timestamp_now, index_x, index_y, index_vx, index_vy)
                    motion2 = self.extract_motion(self.agent_states, id_2, timestamp_now, index_x, index_y, index_vx, index_vy)

                    # Store distance and motion information for the agent pair
                    pair_info[(id_1, id_2)] = {'dis': (dis1, dis2), 'motion1': motion1, 'motion2': motion2}

                # Calculate minimum acceleration and related metrics for the pair group
                a_min, success, solution, min_pet, mean_pet = self.MultiCalculateAcceleration(pair_info, pair_multi, timestamp_now)

                agent_acceleration = {agent_id: acceleration for agent_id, acceleration in zip(pair_multi, solution[:len(pair_multi)])}
                # Store the solution (accelerations) for the current group
                pair_len = len(pair_multi)
                solution_tuple = tuple(solution[:pair_len])
                current_a_min_dict[pair_multi] = (a_min, solution_tuple, min_pet, mean_pet)
                
                for pair in pair_info.keys():
                    if pair not in sub_pair_value: 
                        sub_pair_value[pair] = [0] * len(self.all_timesteps)
                    sum_of_acceleration = sum(abs(agent_acceleration[agent_id]) for agent_id in pair)
                    if sum_of_acceleration > 0.01:
                        sub_pair_value[pair][timestamp_now] = sum_of_acceleration


            # Save the acceleration information for the current timestamp
            timestamp_a[timestamp_now] = current_a_min_dict
        return timestamp_a, sub_pair_value
