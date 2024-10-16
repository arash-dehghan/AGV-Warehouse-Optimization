import sys
sys.dont_write_bytecode = True
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse
import pickle
from Environment import Environment
from CentralAgent import CentralAgent
from ValueFunction import NeurADP
from LearningAgent import LearningAgent
from Experience import Experience
from ResultCollector import ResultCollector
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# import torch
from RequestOrder import RequestOrder

def ensure_batteries(agents):
	for agent in agents:
		assert agent.battery_percentage >= 0
		if agent.battery_percentage == 0:
			assert (agent.next_location  in envt.node_types['C']) and (agent.time_to_next_location == 0.0)

def run_epoch(envt, central_agent, value_function, requests, request_generator, agents_predefined, is_training = False):
	envt.current_time = envt.start_epoch
	ts = int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length)
	Experience.envt = envt
	agents = deepcopy(agents_predefined)
	graph_seen, graph_served = [], []
	global_order_id, total_orders_served, total_orders_seen, time_until_deliveries = 0, 0, 0, []
	past_num_robots_charging, past_avg_robot_battery_percentage, past_avg_robot_capacity, past_avg_human_capacity, past_num_human_only_orders, past_num_both_orders = 0, 0, 0, 0, 0, 0
	graph_seen, graph_num_human_only_orders, graph_num_both_orders, graph_served, graph_served_by_human, graph_served_by_robot, graph_avg_feasible_human, graph_avg_feasible_robot, graph_num_robots_charging, graph_avg_robot_battery_percentage, graph_avg_robot_capacity, graph_avg_human_capacity, graph_matching_sizes = ([] for _ in range(13))
	
	for t in range(ts):
		# print('================')
		# print(f'Iteration: {t}')
		i = 0
		# print(f'State of Agent: (ID: {agents[i].id}), (Is Human: {agents[i].is_human}), (Next Location: {agents[i].next_location}), (Time to Next Location: {agents[i].time_to_next_location}), (Battery Level: {agents[i].battery_percentage}), (# of Assigned Orders: {agents[i].capacity}), (Orders Picked Up: {agents[i].orders_picked_up}), (Orders to Pick Up: {agents[i].orders_to_pickup})')
		ensure_batteries(agents)
		# Generate and add deadlines to new orders, add new orders to remaining orders
		current_orders = [central_agent.set_deadlines(order, i) for i,order in enumerate(requests.get(t,[]), start=global_order_id)]

		# print(f'Incoming Orders: {current_orders}')
		global_order_id += len(current_orders)
		current_order_ids = [order.id for order in current_orders]

		# Get feasible actions for each agent
		feasible_actions = central_agent.get_feasible_actions(agents, current_orders)
		# print(f'Number of Feasible Actions: R={sum([1 for act in feasible_actions[0] if act[1][0] == "R"])}, C={sum([1 for act in feasible_actions[0] if act[1][0] == "C"])}, O={sum([1 for act in feasible_actions[0] if act[1][0] == "O"])}')

		# Get other external info to add to post-decision state
		num_robots_charging, avg_robot_battery_percentage, avg_robot_capacity, avg_human_capacity, num_human_only_orders, num_both_orders = central_agent.get_external_infor(agents, current_orders)

		# Create Experience
		experience = Experience(deepcopy(agents), current_order_ids, feasible_actions, envt.current_time, num_robots_charging, avg_robot_battery_percentage, avg_robot_capacity, avg_human_capacity, num_human_only_orders, num_both_orders, past_num_robots_charging, past_avg_robot_battery_percentage, past_avg_robot_capacity, past_avg_human_capacity, past_num_human_only_orders, past_num_both_orders)

		# Score the feasible actions and pair taking action with its score
		scored_actions_all_agents = value_function.get_value([experience])
		final_pairings, id_to_pairings = value_function.pair_scores(scored_actions_all_agents,feasible_actions)

		# Choose actions for each agent
		matchings, scores = central_agent.choose_actions(final_pairings, id_to_pairings, len(agents), current_order_ids, is_training=is_training)

		if matchings[0][1] in ['R_90', 'R_9']:
			print(envt.rebalancing_nodes)
			exit()
		# assert matchings[0][1] not in ['R_90', 'R_9']
		# print(f'Matched Action: {matchings[i]}')
		# print(f'Score of Matched Action: {scores[i]}')

		# Update if training
		if is_training:
			if t > 0:
				# Update replay buffer
				value_function.remember(experience)
			# Update value function every TRAINING_FREQUENCY timesteps
			if ((int(envt.current_time) / int(envt.epoch_length)) % 1 == 0):
				value_function.update(central_agent)

		# Set the new trajectories for each agent
		orders_served, time_until_return_values, both_served, human_only_served, served_by_human = central_agent.set_new_paths(agents, matchings)
		# print(f'Orders Accepted: {orders_served}')
		# print(f'Orders Dropped Off: {both_served + human_only_served}')
		total_orders_served += orders_served
		total_orders_seen += len(current_orders)
		time_until_deliveries += time_until_return_values

		if not is_training:
			graph_seen.append(len(current_orders)) # Number of orders seen at time t
			graph_num_human_only_orders.append(num_human_only_orders)
			graph_num_both_orders.append(num_both_orders)
			graph_served.append(orders_served) # Number of orders served at time t
			graph_served_by_human.append(served_by_human) # Number of orders served by humans at time t
			graph_served_by_robot.append(orders_served - served_by_human) # Number of orders served by robots at time t
			graph_avg_feasible_human.append(np.mean([sum([1 for b in feasible_actions[a] if (b[1][0] == 'O')]) for a in range(len(agents)) if agents[a].is_human]) if envt.num_humans > 0 else 0) # Avg # of feasible order matching actions for humans at time t
			graph_avg_feasible_robot.append(np.mean([sum([1 for b in feasible_actions[a] if (b[1][0] == 'O')]) for a in range(len(agents)) if not agents[a].is_human]) if envt.num_robots > 0 else 0) # Avg # of feasible order matching actions for robots at time t
			graph_num_robots_charging.append(num_robots_charging) # Number of robots charging at time t
			graph_avg_robot_battery_percentage.append(avg_robot_battery_percentage) # Avg battery percentage for robots at time t
			graph_avg_robot_capacity.append(avg_robot_capacity)
			graph_avg_human_capacity.append(avg_human_capacity)
			m_sizes = [central_agent._get_number_orders_served(match[1]) for match in matchings.values() if match[1][0] == 'O']
			graph_matching_sizes.append(np.mean(m_sizes) if len(m_sizes) > 0 else np.nan)

		# Update the time
		envt.current_time += envt.epoch_length
		past_num_robots_charging = num_robots_charging
		past_avg_robot_battery_percentage = avg_robot_battery_percentage
		past_avg_robot_capacity = avg_robot_capacity
		past_avg_human_capacity = avg_human_capacity
		past_num_human_only_orders = num_human_only_orders
		past_num_both_orders = num_both_orders

	if is_training:
		envt.num_days_trained += 1
	graph_avg_delivery_time = [] if is_training else np.array([np.mean(time_until_deliveries) for _ in range(ts)])

	return total_orders_served, total_orders_seen, np.array([graph_seen, graph_num_human_only_orders, graph_num_both_orders, graph_served, graph_served_by_human, graph_served_by_robot, graph_avg_feasible_human, graph_avg_feasible_robot, graph_num_robots_charging, graph_avg_robot_battery_percentage, graph_avg_robot_capacity, graph_avg_human_capacity, graph_matching_sizes, graph_avg_delivery_time])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_humans', '--num_humans', type=int, default=5)
	parser.add_argument('-num_robots', '--num_robots', type=int, default=5)
	parser.add_argument('-battery_rate', '--battery_rate', type=int, default=0.5)
	parser.add_argument('-charging_rate', '--charging_rate', type=int, default=5)
	parser.add_argument('-num_rows', '--num_rows', type=int, default=10)
	parser.add_argument('-num_cols', '--num_cols', type=int, default=10)
	parser.add_argument('-num_cs', '--num_cs', type=int , default=2)
	parser.add_argument('-horizon_length', '--horizon_length', type=int, default=1440)
	parser.add_argument('-edge_travel_time_human', '--edge_travel_time_human', type=float, default=0.5)
	parser.add_argument('-edge_travel_time_robot', '--edge_travel_time_robot', type=float, default=0.5)
	parser.add_argument('-remove_vert', '--remove_vert', type=int, default=1)
	parser.add_argument('-epoch_length', '--epoch_length', type=int , default=5)
	parser.add_argument('-order_multiplier', '--order_multiplier', type=int, default=5)
	parser.add_argument('-percentage_only_human_orders', '--percentage_only_human_orders', type=float, default=0.0)
	parser.add_argument('-dt', '--delaytime', type=float, default=17.5)
	parser.add_argument('-human_capacity', '--human_capacity', type=float, default=2)
	parser.add_argument('-robot_capacity', '--robot_capacity', type=float, default=2)
	parser.add_argument('-rebalancing_allowed', '--rebalancing_allowed', type=float, default=1)
	parser.add_argument('-train_days', '--train_days', type=int, default=60)
	parser.add_argument('-test_days', '--test_days', type=int, default=5)
	parser.add_argument('-test_every', '--test_every', type=int, default=5)
	parser.add_argument('-seed', '--seed', type=int , default=1)
	args = parser.parse_args()
	args.numagents, args.battery_reduction_epoch = (args.num_humans + args.num_robots), (args.battery_rate * args.epoch_length)

	filename = f'{args.num_rows * args.num_cols}_{args.num_cs}_{args.remove_vert}_{args.horizon_length}_{args.edge_travel_time_human}_{args.edge_travel_time_robot}_{args.epoch_length}_{args.order_multiplier}_{args.percentage_only_human_orders}_{args.seed}'
	request_generator = pickle.load(open(f'../data/generations/{filename}/data_{filename}.pickle','rb'))
	envt = Environment(args.num_humans, args.num_robots, args.epoch_length, args.horizon_length, args.edge_travel_time_human, args.edge_travel_time_robot, args.remove_vert, request_generator.node_types, request_generator.rebalancing_nodes, args.human_capacity, args.robot_capacity, args.battery_rate, args.charging_rate, args.delaytime)
	central_agent = CentralAgent(envt, args.num_humans, args.num_robots, args.delaytime, args.rebalancing_allowed)
	value_function = NeurADP(envt, filename)

	#############################################################################
	#### SETTING #1: Learn to reject most immediate orders for more later on ####
	#############################################################################
	# toy_orders = {0 : [RequestOrder(22, 0 , 0, 0, 1)], 4: [RequestOrder(66, 0, 4, 0, 1), RequestOrder(76, 0, 4, 0, 1)]}
	# toy_agents = [LearningAgent(0, True, 36, 100, 0)]

	#############################################################################
	#### SETTING #2: Learn to charge instead of serve orders immediately ####
	#############################################################################
	# toy_orders = {0 : [RequestOrder(17, 0 , 0, 0, 1)], 4: [RequestOrder(66, 0, 4, 0, 1), RequestOrder(76, 0, 4, 0, 1)]}
	# toy_agents = [LearningAgent(0, False, 9, 25, 0)]

	#############################################################################
	#### SETTING #3: Learn to match best (Ex: not just humans first) ####
	#############################################################################
	toy_orders = {0 : [RequestOrder(71, 0 , 0, 0, 1), RequestOrder(81, 0 , 0, 0, 1)], 1: [RequestOrder(23, 0, 1, 0, 1), RequestOrder(33, 0, 1, 0, 1), RequestOrder(14, 0, 1, 0, 1), RequestOrder(24, 0, 1, 0, 1)]}
	toy_agents = [LearningAgent(0, True, 13, 100, 0), LearningAgent(1, False, 81, 100, 0)]

	### TESTING ###
	served, seen, _ , _ = run_epoch(envt, central_agent, value_function, deepcopy(toy_orders), None, deepcopy(toy_agents), False)
	print(served, seen)

	exit()

	### TRAINING ####
	for _ in tqdm(range(args.train_days)):
		served, seen, _ , _ = run_epoch(envt, central_agent, value_function, deepcopy(toy_orders), None, deepcopy(toy_agents), True)
		print(served, seen)


	exit()





