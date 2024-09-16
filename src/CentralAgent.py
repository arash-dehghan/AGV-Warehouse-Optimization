from collections import Counter
from itertools import combinations, permutations
import docplex.mp.model as cpx
import cplex
from copy import deepcopy
import numpy as np

class CentralAgent(object):
	def __init__(self, envt, num_humans, num_robots, delay_allowed, rebalancing_allowed):
		self.envt = envt
		self.num_humans = num_humans
		self.num_robots = num_robots
		self.delay_allowed = delay_allowed
		self.rebalancing_allowed = rebalancing_allowed

	def set_deadlines(self, order, order_id):
		order.deadline = self.envt.current_time + self.delay_allowed
		order.origin_time *= self.envt.epoch_length
		order.id = order_id
		order.update_state(self.envt.current_time)
		return order

	# def _check_break_status(self, agent):
	# 	return (self.envt.current_time < agent.shift_start) or (self.envt.current_time >= agent.shift_end)

	# def _check_at_warehouse_status(self, agent):
	# 	return (not agent.time_until_return) and (not self._check_break_status(agent))

	def get_external_infor(self, agents, orders):
		# Number of robots charging
		num_robots_charging = 0 if self.num_robots == 0 else len([1 for agent in agents if (not agent.is_human) and (self.envt.node_types['D'][0] == agent.next_location) and (agent.time_to_next_location == 0.0)])
		# Average robot battery percentage
		avg_robot_battery_percentage = 0.0 if self.num_robots == 0 else sum([agent.battery_percentage for agent in agents if (not agent.is_human)]) / self.num_robots
		# Average capacity of robots
		avg_robot_capacity = -1.0 if self.num_robots == 0 else sum([agent.capacity for agent in agents if not agent.is_human]) / self.num_robots
		# Average capacity of humans
		avg_human_capacity = -1.0 if self.num_humans == 0 else sum([agent.capacity for agent in agents if agent.is_human]) / self.num_humans
		# Number of human only orders
		num_human_only_orders = sum([1 for order in orders if order.human_only_order])
		# Number of human and robot available orders
		num_both_orders = len(orders) - num_human_only_orders
		return num_robots_charging, avg_robot_battery_percentage, avg_robot_capacity, avg_human_capacity, num_human_only_orders, num_both_orders

	def get_feasible_actions(self, agents, orders):
		potential_actions, agent_feasible_actions = self._get_potential_actions(orders), []
		for agent in agents:
			agent_charging_actions = [] if agent.is_human else self._get_charging_actions(agent)
			agent_rebalancing_actions = self._get_rebalancing_actions(agent)
			agent_order_matching_actions = self._get_order_matchings(potential_actions, agent)
			if agent.capacity > 0:
				agent_order_matching_actions += [([],'O_',0)]
			all_agent_actions = agent_charging_actions + agent_rebalancing_actions + agent_order_matching_actions
			agent_feasible_actions.append(all_agent_actions)
		return agent_feasible_actions

	def _get_charging_actions(self, agent):
		if agent.capacity == 0:
			# Get the closest charging station to the robot
			nearest_cs = self.envt.get_nearest_charging_station(agent.next_location)
			# Calculate the amount of battery percentage used to get to that charging station
			battery_used = self.envt.calculate_battery_expenditure(agent.next_location, agent.time_to_next_location, nearest_cs)
			# Ensure robot has enough battery to get there
			assert len(agent.orders_to_pickup) == 0 and len(agent.orders_picked_up) == 0
			if agent.battery_percentage < battery_used:
				print(agent.id)
				print(agent.battery_percentage)
				print(battery_used)
				print(agent.state)
				exit()
			assert agent.battery_percentage >= battery_used
			return [(nearest_cs,f'C_{nearest_cs}')]
		return []

	def _get_rebalancing_actions(self, agent):
		balancing_locations = []
		if agent.capacity == 0:
			assert len(agent.orders_to_pickup) == 0 and len(agent.orders_picked_up) == 0
			if self.rebalancing_allowed:
				rebalancing_points = self.envt.rebalancing_nodes if agent.next_location in self.envt.node_types['C'] else self.envt.rebalancing_nodes + [agent.next_location]
				for rebalancing_location in rebalancing_points:
					if agent.is_human:
						balancing_locations.append((rebalancing_location,f'R_{rebalancing_location}'))
					else:
						# Get the location information of where the robot would be if they rebalanced towards the rebalancing location
						location, time_to_location = self.envt.get_rebalancing_next_location(agent.next_location, agent.time_to_next_location, rebalancing_location, agent.is_human)
						# Now ensure from that location in the next time-step they'd still have enough battery to reach a charging station if needed
						nearest_cs = self.envt.get_nearest_charging_station(location)
						battery_used = self.envt.calculate_battery_expenditure(location, time_to_location, nearest_cs)
						if agent.battery_percentage >= battery_used + (self.envt.epoch_length * self.envt.battery_rate):
							balancing_locations.append((rebalancing_location,f'R_{rebalancing_location}'))
			else:
				if agent.is_human:
					balancing_locations.append((agent.next_location, f'R_{agent.next_location}'))
				else:
					if agent.next_location not in self.envt.node_types['C']:
						location, time_to_location = self.envt.get_rebalancing_next_location(agent.next_location, agent.time_to_next_location, agent.next_location, agent.is_human)
						nearest_cs = self.envt.get_nearest_charging_station(location)
						battery_used = self.envt.calculate_battery_expenditure(location, time_to_location, nearest_cs)
						if agent.battery_percentage >= battery_used + (self.envt.epoch_length * self.envt.battery_rate):
							balancing_locations.append((agent.next_location,f'R_{agent.next_location}'))
		return balancing_locations

	def _get_order_matchings(self, potential_actions ,agent):
		agents_actions = []
		# First, check that the agent has enough capacity to take on any more orders
		capacity_available = (self.envt.human_capacity - agent.capacity) if agent.is_human else (self.envt.robot_capacity - agent.capacity)
		if capacity_available > 0:
			for cap in range(1, capacity_available + 1):
				for act in potential_actions[cap]:
					# Second, check that if the agent is a robot, they are able to handle all of the items in the assignment
					if (not agent.is_human) and (not self._check_robot_orders_handlable(act)): continue
					# Third, check if there is any feasible way for the agent to be assigned this new batch of orders
					new_to_pickup_ordering, dropoff_delay = self._check_feasibility(agent, act)
					if len(new_to_pickup_ordering) > 0:
						agents_actions.append((new_to_pickup_ordering, 'O_' + '_'.join([str(o.id) for o in act]), dropoff_delay))
		return agents_actions

	def _get_potential_actions(self, orders):
		return {size : [list(action) for action in list(combinations(orders, size))] for size in range(1, max(self.envt.human_capacity, self.envt.robot_capacity) + 1)}

	def _check_robot_orders_handlable(self, act):
		return not any(order.human_only_order for order in act)

	def _find_next_decision_epoch(self, time):
		times = list(range(self.envt.current_time, self.envt.stop_epoch + 2 * self.envt.epoch_length, self.envt.epoch_length))
		for t in times:
			if time <= t:
				return t
		print("OH NO, ERROR!")
		print(time)
		exit()

	def _check_feasibility(self, agent, action):
		all_orders = agent.orders_picked_up + agent.orders_to_pickup + action
		orders = agent.orders_to_pickup + action
		all_pickups = set([order.pickup for order in orders])
		best_ordering, best_dropoff_time = [], self.envt.stop_epoch + 1

		for ordering in permutations(all_pickups):
			time = self.envt.current_time + agent.time_to_next_location
			full_ordering = [agent.next_location] + list(ordering) + self.envt.node_types['D']
			location_arrival_times = {}

			for i in range(len(full_ordering) - 1):
				time += self.envt.get_travel_time(full_ordering[i], full_ordering[i + 1], agent.is_human)
				location_arrival_times[full_ordering[i + 1]] = time

			if all(order.deadline >= location_arrival_times[self.envt.node_types['D'][0]] for order in all_orders):
					if location_arrival_times[self.envt.node_types['D'][0]] < best_dropoff_time:
						best_ordering = list(ordering)
						best_dropoff_time = location_arrival_times[self.envt.node_types['D'][0]]

		final_ordering = [order for loc in best_ordering for order in orders if order.pickup == loc]

		if (len(final_ordering) > 0) and (not agent.is_human):
			nearest_cs = self.envt.get_nearest_charging_station(self.envt.node_types['D'][0])
			next_time = self._find_next_decision_epoch(best_dropoff_time)
			next_possible_charging_time = next_time + self.envt.get_travel_time(self.envt.node_types['D'][0], nearest_cs, False)
			time_until_next_charging = next_possible_charging_time - self.envt.current_time
			battery_used = self.envt.battery_rate * time_until_next_charging
			if agent.battery_percentage < battery_used:
				return [],-1

		return [order for loc in best_ordering for order in orders if order.pickup == loc], (best_dropoff_time - self.envt.current_time + agent.time_to_next_location)

	def choose_actions(self, agent_action_choices, id_to_pairings, num_agents, request_ids, is_training):
		#############
		### MODEL ###
		model = cpx.Model(name="Matching Model")

		# ### VARIABLES ###
		variables = [f'x-{a}-{b}' for a in agent_action_choices.keys() for b in agent_action_choices[a].keys()]
		x_a_d = {(a,b): model.binary_var(name=f'x-{a}-{b}') for a in agent_action_choices.keys() for b in agent_action_choices[a].keys()}

		### CONSTRAINTS ###
		flow_driver_conservation_const = {a : model.add_constraint( ct=(model.sum(x_a_d[(a,b)] for b in agent_action_choices[a]) == 1), ctname=f'constraint_a_{a}' ) for a in range(num_agents)}
		flow_order_conservation_const = {order : model.add_constraint(ct=(model.sum(x_a_d[(a,b)] for a in range(num_agents) for b in agent_action_choices[a] if self._contains_order(order, b)) <= 1), ctname=f'constraint_o_{order}') for order in request_ids}

		### OBJECTIVE ###
		total_obj = model.sum(x_a_d[(a,b)] * (agent_action_choices[a][b] + self._get_noise(x_a_d[(a,b)], is_training)) for a in agent_action_choices.keys() for b in agent_action_choices[a].keys())
		model.set_objective('max', total_obj)

		### SOLVE ###
		solution = model.solve()
		assert solution
		#############

		final_actions = {}
		final_scores = {}
		for a in agent_action_choices.keys():
			for b in agent_action_choices[a].keys():
				if solution.get_value(f'x-{a}-{b}') > 0:
					final_actions[a] = id_to_pairings[a][b]
					final_scores[a] = agent_action_choices[a][b]

		return final_actions, final_scores

	def _contains_order(self, order_id, action):
		action_description = action.split('_')
		return (str(order_id) in action_description) if action_description[0] == 'O' else False

	def _get_noise(self, variable, is_training):
		variable_description = variable.get_name().split('-')[-1].split('_')
		if variable_description[0] == 'O':
			variable_description = [i for i in variable_description[1:] if i != '']
			fulfills_orders = True if len(variable_description) > 0 else False
		else:
			fulfills_orders = False
		stdev = 1 + (4000 if not fulfills_orders == '_' else 1000) / ((self.envt.num_days_trained + 1) * (self.envt.num_humans + self.num_robots))
		return abs(np.random.normal(0, stdev)) if is_training else 0

	def set_new_paths(self, agents, matchings):
		served, served_by_human, both_served, human_only_served, time_until_dropoffs = 0, 0, 0, 0, []
		for i in range(len(agents)):
			agent = agents[i]
			action = matchings[i]
			action_type = action[1][0]
			if action_type == 'R':
				self.envt.simulate_rebalancing(agent, action[0], self.envt.current_time)
			elif action_type == 'C':
				self.envt.simulate_charging(agent, action[0], self.envt.current_time)
			elif action_type == 'O':
				time_until_dropoff, both_delivered, human_only_delivered = self.envt.simulate_order_matching(agent, action[0], self.envt.current_time)
				for t in time_until_dropoff:
					assert t <= self.envt.delaytime
				time_until_dropoffs += time_until_dropoff
				num_orders_served_by_act = self._get_number_orders_served(action[1])
				served += num_orders_served_by_act
				both_served += both_delivered
				human_only_served += human_only_delivered
				if agent.is_human:
					served_by_human += num_orders_served_by_act
					
		# exit()
		return served, time_until_dropoffs, both_served, human_only_served, served_by_human

	def _get_number_orders_served(self, label):
		return sum([1 for l in label.split('_')[1:] if l != ''])


