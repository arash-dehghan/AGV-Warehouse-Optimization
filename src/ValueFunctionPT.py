from copy import deepcopy
import time
import pickle
import numpy as np
from ReplayBuffer import SimpleReplayBuffer, PrioritizedReplayBuffer
from os.path import isfile, isdir

from RequestOrder import RequestOrder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('GPU utilized' if torch.cuda.is_available() else 'CPU utilized')

class NNModel(nn.Module):
	def __init__(self, envt):
		super(NNModel, self).__init__()
		self.envt = envt
		# Embedding for locations
		self.location_embedding = nn.Embedding(
			num_embeddings=self.envt.num_locations + 1,
			embedding_dim=100,
			padding_idx=0,
		)
		# LSTM for path embeddings
		self.lstm = nn.LSTM(
			input_size=100 + 1, hidden_size=200, batch_first=True, bidirectional=False
		)
		# Time embedding
		self.time_embedding = nn.Sequential(
			nn.Linear(1, 100),
			nn.ELU(),
		)
		# State embedding
		self.state_embed = nn.Sequential(
			nn.Linear(200 + 100 + 9, 300),  # Corrected here
			nn.ELU(),
			nn.Linear(300, 300),
			nn.ELU(),
		)
		# Output layer
		self.output_layer = nn.Linear(300, 1)

	def forward(
		self,
		locations_input,
		delays_input,
		current_time_input,
		capacity_input,
		is_human_input,
		battery_percentage_input,
		num_robots_charging_input,
		avg_robot_battery_percentage_input,
		avg_robot_capacity_input,
		avg_human_capacity_input,
		num_human_only_orders_input,
		num_both_orders_input,
	):
		# Embedding lookup for locations
		path_location_embed = self.location_embedding(locations_input)
		# Concatenate embeddings and delays
		delays_input = delays_input.squeeze(2)  # Remove the last dimension
		path_input = torch.cat([path_location_embed, delays_input.unsqueeze(2)], dim=2)
		# Reverse the sequence (go_backwards=True)
		path_input_reversed = torch.flip(path_input, dims=[1])
		# Apply LSTM
		_, (h_n, _) = self.lstm(path_input_reversed)
		path_embed = h_n.squeeze(0)
		# Time embedding
		current_time_embed = self.time_embedding(current_time_input)
		# Other inputs
		other_inputs = torch.cat(
			[
				capacity_input,
				is_human_input,
				battery_percentage_input,
				num_robots_charging_input,
				avg_robot_battery_percentage_input,
				avg_robot_capacity_input,
				avg_human_capacity_input,
				num_human_only_orders_input,
				num_both_orders_input,
			],
			dim=1,
		)
		# State embedding
		state_embed_input = torch.cat(
			[path_embed, current_time_embed, other_inputs], dim=1
		)
		state_embed = self.state_embed(state_embed_input)
		# Output
		output = self.output_layer(state_embed)
		return output


class NeurADP:
	def __init__(self, envt, emb_file_name, load_model_loc=""):
		self.envt = envt
		self.emb_file_name = emb_file_name
		self.M = self.envt.delaytime * self.envt.car_capacity

		self.gamma = 0.9
		self.batch_size_fit = 32  # Number of samples per batch to use during fitting
		self.batch_size_predict = (
			32  # Number of samples per batch to use during prediction
		)
		self.target_update_tau = 0.001
		self.num_min_train_samples = (
			1000  # Minimum size of replay buffer needed to begin sampling
		)
		self.num_samples = 50

		# Get Replay Buffer
		min_len_replay_buffer = 1e6 / (
			self.envt.num_humans + self.envt.num_robots
		)  # What is the size of the replay buffer???
		epochs_in_episode = (
			self.envt.stop_epoch - self.envt.start_epoch
		) / self.envt.epoch_length
		len_replay_buffer = max((min_len_replay_buffer, epochs_in_episode))
		self.replay_buffer = PrioritizedReplayBuffer(MAX_LEN=int(len_replay_buffer))

		# Get NN Model
		self.model = torch.load(load_model_loc) if load_model_loc else self._init_NN()
		self.model.to(device)

		# Define Loss and Optimizer
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.loss_fn = nn.MSELoss(reduction="none")

		# Get target-NN
		self.target_model = self._init_NN()
		self.target_model.load_state_dict(self.model.state_dict())
		self.target_model.to(device)

	def _init_NN(self):
		"""
		Neural network structure
		"""
		return NNModel(self.envt)

	def get_value(self, experiences, network=None):
		# Format experiences to input into the NN
		action_inputs_all_agents, shape_info = self._format_experiences(
			experiences, is_current=False
		)

		# Prepare inputs as tensors
		inputs = {}
		for key, value in action_inputs_all_agents.items():
			if key == "locations_input":
				inputs[key] = torch.from_numpy(value).long().to(device)
			elif key == "delays_input":
				inputs[key] = torch.from_numpy(value).float().to(device)
			else:
				inputs[key] = torch.from_numpy(value).float().to(device)

		# Set model to evaluation mode
		self.model.eval()
		if network is None:
			model = self.model
		else:
			model = network

		with torch.no_grad():
			# Compute expected future values
			output = model(
				inputs["locations_input"],
				inputs["delays_input"],
				inputs["current_time_input"],
				inputs["capacity_input"],
				inputs["is_human_input"],
				inputs["battery_percentage_input"],
				inputs["num_robots_charging_input"],
				inputs["avg_robot_battery_percentage_input"],
				inputs["avg_robot_capacity_input"],
				inputs["avg_human_capacity_input"],
				inputs["num_human_only_orders_input"],
				inputs["num_both_orders_input"],
			)

		expected_future_values_all_agents = output.cpu().detach().numpy()

		# Format output back into the shape of a 2D list (for each agent and their corresponding actions)
		expected_future_values_all_agents = self._reconstruct_NN_output(
			expected_future_values_all_agents, shape_info
		)

		# Format the feasible actions into the same 2D list with the same shape
		feasible_actions_all_agents = [
			feasible_actions
			for experience in experiences
			for feasible_actions in experience.feasible_actions_all_agents
		]

		# Empty list to store all the scores for each agent and their feasible actions
		scored_actions_all_agents = []
		for expected_future_values, feasible_actions in zip(
			expected_future_values_all_agents, feasible_actions_all_agents
		):
			scored_actions = [
				self._get_score(action, value)
				for action, value in zip(feasible_actions, expected_future_values)
			]
			scored_actions_all_agents.append(scored_actions)

		return scored_actions_all_agents

	def _format_experiences(self, experiences, is_current):
		action_inputs_all_agents = None
		for experience in experiences:
			# If experience hasn't been formatted, format it. Set the experience representation to the formatted and normalized post-decision state data
			if not (self.__class__.__name__ in experience.representation):
				experience.representation[
					self.__class__.__name__
				] = self._get_input_batch_next_state(experience)
			if is_current:
				batch_input = self._format_input_batch(
					[[agent] for agent in experience.agents],
					experience.time,
					experience.past_num_robots_charging,
					experience.past_avg_robot_battery_percentage,
					experience.past_avg_robot_capacity,
					experience.past_avg_human_capacity,
					experience.past_num_human_only_orders,
					experience.past_num_both_orders,
				)
			else:
				batch_input = deepcopy(experience.representation[self.__class__.__name__])

			if action_inputs_all_agents is None:
				action_inputs_all_agents = batch_input

		assert action_inputs_all_agents is not None

		return self._flatten_NN_input(action_inputs_all_agents)

	def _get_input_batch_next_state(self, experience):
		all_agents_post_actions = []
		for agent, feasible_actions in zip(
			experience.agents, experience.feasible_actions_all_agents
		):
			agents_post_actions = []
			assert agent.time_to_next_location < self.envt.epoch_length
			for action in feasible_actions:
				agent_next_time = deepcopy(agent)
				action_type = action[1][0]
				if action_type == "R":
					self.envt.simulate_rebalancing(
						agent_next_time, action[0], experience.time
					)
				elif action_type == "C":
					self.envt.simulate_charging(
						agent_next_time, action[0], experience.time
					)
				elif action_type == "O":
					_, _, _ = self.envt.simulate_order_matching(
						agent_next_time, action[0], experience.time
					)
				else:
					print(agent.is_human)
					print(action_type)
					print("ERROR")
					exit()
				agents_post_actions.append(agent_next_time)
			assert len(feasible_actions) == len(agents_post_actions)
			all_agents_post_actions.append(agents_post_actions)

		# Update the time to the next epoch time
		next_time = experience.time + self.envt.epoch_length

		# Return formatted inputs of these agents
		return self._format_input_batch(
			all_agents_post_actions,
			next_time,
			experience.num_robots_charging,
			experience.avg_robot_battery_percentage,
			experience.avg_robot_capacity,
			experience.avg_human_capacity,
			experience.num_human_only_orders,
			experience.num_both_orders,
		)

	def _format_input_batch(
		self,
		all_agents_post_actions,
		current_time,
		num_robots_charging,
		avg_robot_battery_percentage,
		avg_robot_capacity,
		avg_human_capacity,
		num_human_only_orders,
		num_both_orders,
	):
		# Create empty dictionary for all the respective NN input values for each possible feasible action for each agent
		inp = {
			"locations_input": [],
			"delays_input": [],
			"current_time_input": [],
			"capacity_input": [],
			"is_human_input": [],
			"battery_percentage_input": [],
			"num_robots_charging_input": [],
			"avg_robot_battery_percentage_input": [],
			"avg_robot_capacity_input": [],
			"avg_human_capacity_input": [],
			"num_human_only_orders_input": [],
			"num_both_orders_input": [],
		}
		# For specific agent and their respective list of post-decision agent objects
		for agent_post_actions in all_agents_post_actions:
			# Initialize empty lists of post-decision state values to input into NN
			(
				locations_inputs,
				delays_inputs,
				current_time_inputs,
				capacity_inputs,
				is_human_inputs,
				battery_percentage_inputs,
				num_robots_charging_inputs,
				avg_robot_battery_percentage_inputs,
				avg_robot_capacity_inputs,
				avg_human_capacity_inputs,
				num_human_only_orders_inputs,
				num_both_orders_inputs,
			) = ([] for i in range(12))

			# For each post-decision state agent object for agent i
			for agent in agent_post_actions:
				# Get formatted output for the state
				(
					locations_input,
					delays_input,
					current_time_input,
					capacity_input,
					is_human_input,
					battery_percentage_input,
					num_robots_charging_input,
					avg_robot_battery_percentage_input,
					avg_robot_capacity_input,
					avg_human_capacity_input,
					num_human_only_orders_input,
					num_both_orders_input,
				) = self._format_input(
					agent,
					current_time,
					num_robots_charging,
					avg_robot_battery_percentage,
					avg_robot_capacity,
					avg_human_capacity,
					num_human_only_orders,
					num_both_orders,
				)

				locations_inputs.append(locations_input)
				delays_inputs.append(delays_input)
				current_time_inputs.append([current_time_input])
				capacity_inputs.append([capacity_input])
				is_human_inputs.append([is_human_input])
				battery_percentage_inputs.append([battery_percentage_input])
				num_robots_charging_inputs.append([num_robots_charging_input])
				avg_robot_battery_percentage_inputs.append(
					[avg_robot_battery_percentage_input]
				)
				avg_robot_capacity_inputs.append([avg_robot_capacity_input])
				avg_human_capacity_inputs.append([avg_human_capacity_input])
				num_human_only_orders_inputs.append([num_human_only_orders_input])
				num_both_orders_inputs.append([num_both_orders_input])

			inp["locations_input"].append(locations_inputs)
			inp["delays_input"].append(delays_inputs)
			inp["current_time_input"].append(current_time_inputs)
			inp["capacity_input"].append(capacity_inputs)
			inp["is_human_input"].append(is_human_inputs)
			inp["battery_percentage_input"].append(battery_percentage_inputs)
			inp["num_robots_charging_input"].append(num_robots_charging_inputs)
			inp["avg_robot_battery_percentage_input"].append(
				avg_robot_battery_percentage_inputs
			)
			inp["avg_robot_capacity_input"].append(avg_robot_capacity_inputs)
			inp["avg_human_capacity_input"].append(avg_human_capacity_inputs)
			inp["num_human_only_orders_input"].append(num_human_only_orders_inputs)
			inp["num_both_orders_input"].append(num_both_orders_inputs)

		return inp

	def _format_input(
		self,
		agent,
		current_time,
		num_robots_charging,
		avg_robot_battery_percentage,
		avg_robot_capacity,
		avg_human_capacity,
		num_human_only_orders,
		num_both_orders,
	):
		# Normalizing Inputs
		locations_input = self._add_locations(agent.next_location, agent.orders_to_pickup)
		delays_input = self._add_delays(
			agent.time_to_next_location,
			agent.orders_to_pickup,
			agent.orders_picked_up,
			current_time,
		)
		current_time_input = (current_time - self.envt.start_epoch) / (
			self.envt.stop_epoch - self.envt.start_epoch
		)
		capacity_input = agent.capacity / self.envt.car_capacity
		is_human_input = float(agent.is_human)
		battery_percentage_input = agent.battery_percentage / 100
		num_robots_charging_input = (
			0 if self.envt.num_robots == 0 else num_robots_charging / self.envt.num_robots
		)
		avg_robot_battery_percentage_input = avg_robot_battery_percentage
		avg_robot_capacity_input = avg_robot_capacity
		avg_human_capacity_input = avg_human_capacity
		num_human_only_orders_input = (
			0.0 if num_both_orders == 0 else num_human_only_orders / num_both_orders
		)
		num_both_orders_input = num_both_orders
		return (
			locations_input,
			delays_input,
			current_time_input,
			capacity_input,
			is_human_input,
			battery_percentage_input,
			num_robots_charging_input,
			avg_robot_battery_percentage_input,
			avg_robot_capacity_input,
			avg_human_capacity_input,
			num_human_only_orders_input,
			num_both_orders_input,
		)

	def _add_locations(self, next_location, orders_to_pickup):
		locations_input = np.zeros(
			shape=(self.envt.car_capacity * 2 + 1,), dtype=np.int64
		)
		locations_input[self.envt.car_capacity] = next_location + 1
		for i, order in enumerate(orders_to_pickup):
			j = self.envt.car_capacity + i + 1
			locations_input[j] = order.pickup + 1
		return locations_input

	def _add_delays(
		self, time_to_next_location, orders_to_pickup, orders_picked_up, current_time
	):
		delays = np.zeros(shape=(self.envt.car_capacity * 2 + 1, 1)) - 1
		delays[self.envt.car_capacity] = (
			time_to_next_location / self.envt.delaytime
		)
		for i, order in enumerate(orders_picked_up):
			delays[i] = (order.deadline - current_time) / self.envt.delaytime
		for i, order in enumerate(orders_to_pickup):
			j = i + self.envt.car_capacity + 1
			delays[j] = (order.deadline - current_time) / self.envt.delaytime
		return delays

	def _flatten_NN_input(self, NN_input):
		# Our input here, NN_input, is a dictionary where each key is an input into the NN (i.e current time, num nearby agents, etc.)
		# Within each key is a 2D list, the first index of the 2D list represents a given agent (say agent 0)
		# The second index represents the value if agent 0 had taken action i and been in a given post-decision state
		# In the loop below we just want to document the shape of the 2D list for just the first element of the dictionary (since all the others are the same shape)
		shape_info = []
		for key, value in NN_input.items():
			# Remember the shape information of the inputs
			# If the shape_info list is empty
			if not shape_info:
				cumulative_sum = 0
				shape_info.append(cumulative_sum)
				for idx, list_el in enumerate(value):
					cumulative_sum += len(list_el)
					shape_info.append(cumulative_sum)
			# Reshape the dictionary element of 2D lists into just a 1D list
			NN_input[key] = np.array([element for array in value for element in array])
		# Return the dictionary of 1D lists that has been reshaped and also return the original shape information of the 2D lists
		return NN_input, shape_info

	def _reconstruct_NN_output(self, NN_output, shape_info):
		# Flatten output
		NN_output = NN_output.flatten()

		# Reshape it into the shape it was when it was a 2D list (for each agent and their corresponding actions) and return the 2D list
		assert shape_info
		output_as_list = []
		for idx in range(len(shape_info) - 1):
			start_idx = shape_info[idx]
			end_idx = shape_info[idx + 1]
			list_el = NN_output[start_idx:end_idx].tolist()
			output_as_list.append(list_el)

		return output_as_list

	def _get_score(self, action, value):
		action_description = action[1].split("_")
		if action_description[0] in ["R", "C"]:
			return 0 + self.gamma * value
		else:
			assert action_description[0] == "O"
			delay_of_orders = action[2]
			action_description = [act for act in action_description if act != ""]
			reward = len(action_description) - 1
			return (self.M * reward - delay_of_orders) + self.gamma * value
			# return reward + self.gamma * value

	def pair_scores(self, scored_actions_all_agents, agents_matchings):
		final_pairings = {agent_id: {} for agent_id in range(len(agents_matchings))}
		id_to_pairings = {agent_id: {} for agent_id in range(len(agents_matchings))}
		for agent_id, (action_matchings, matching_scores) in enumerate(
			zip(agents_matchings, scored_actions_all_agents)
		):
			for action, score in zip(action_matchings, matching_scores):
				action_name = action[1]
				final_pairings[agent_id][action_name] = score
				id_to_pairings[agent_id][action_name] = action
		return final_pairings, id_to_pairings

	def remember(self, experience):
		self.replay_buffer.add(experience)

	def update(self, central_agent):
		# Check if replay buffer has enough samples for an update
		if self.num_min_train_samples > len(self.replay_buffer):
			return

		# SAMPLE FROM REPLAY BUFFER
		if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
			# Get beta value
			beta = min(1, 0.4 + 0.6 * (self.envt.num_days_trained / 200.0))
			# Grab some experiences
			experiences, weights, batch_idxes = self.replay_buffer.sample(self.num_samples, beta)
		else:
			experiences = self.replay_buffer.sample(self.num_samples)
			weights = None
			batch_idxes = range(len(experiences))

		# ITERATIVELY UPDATE POLICY BASED ON SAMPLE
		for experience_idx, (experience, batch_idx) in enumerate(zip(experiences, batch_idxes)):
			# Flatten experiences and associate weight of batch with every flattened experience
			if weights is not None:
				# Apply the weight to all agents in the experience
				w = np.array([weights[experience_idx]] * len(experience.agents))
			else:
				w = np.ones(len(experience.agents))

			# GET TD-TARGET
			# Score experiences
			scored_actions_all_agents = self.get_value([experience], network=self.target_model)

			final_pairings, id_to_pairings = self.pair_scores(
				scored_actions_all_agents, experience.feasible_actions_all_agents
			)

			# Run ILP on these experiences to get expected value at next time step
			_, scores = central_agent.choose_actions(
				final_pairings,
				id_to_pairings,
				len(experience.agents),
				experience.request_ids,
				is_training=False,
			)

			value_next_state = [scores[i] for i in range(len(scores))]

			supervised_targets = np.array(value_next_state).reshape((-1, 1))

			# UPDATE NN BASED ON TD-TARGET
			action_inputs_all_agents, _ = self._format_experiences([experience], is_current=True)

			# Prepare tensors
			inputs = {}
			for key, value in action_inputs_all_agents.items():
				if key == "locations_input":
					inputs[key] = torch.from_numpy(value).long().to(device)
				elif key == "delays_input":
					inputs[key] = torch.from_numpy(value).float().to(device)
				else:
					inputs[key] = torch.from_numpy(value).float().to(device)
			targets = torch.from_numpy(supervised_targets).float().to(device)
			weights_tensor = torch.from_numpy(w).float().to(device)

			# Set model to train mode
			self.model.train()
			# Zero the gradients
			self.optimizer.zero_grad()
			# Compute output
			output = self.model(
				inputs["locations_input"],
				inputs["delays_input"],
				inputs["current_time_input"],
				inputs["capacity_input"],
				inputs["is_human_input"],
				inputs["battery_percentage_input"],
				inputs["num_robots_charging_input"],
				inputs["avg_robot_battery_percentage_input"],
				inputs["avg_robot_capacity_input"],
				inputs["avg_human_capacity_input"],
				inputs["num_human_only_orders_input"],
				inputs["num_both_orders_input"],
			)
			# Compute loss
			loss = self.loss_fn(output, targets)
			loss = (loss * weights_tensor.unsqueeze(1)).mean()
			# Backpropagate
			loss.backward()
			# Optimizer step
			self.optimizer.step()

			# Update priorities of replay buffer after update
			if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
				# Calculate new squared error for this experience
				with torch.no_grad():
					predicted_values = output.detach()
					loss_per_sample = ((predicted_values - targets) ** 2 + 1e-6).cpu().numpy()
					# Average loss over all agents in the experience
					loss_per_experience = loss_per_sample.mean()
				# Update priorities
				self.replay_buffer.update_priorities([batch_idx], [loss_per_experience])

			# Soft update target_model based on the learned model
			self.soft_update(self.target_model, self.model, self.target_update_tau)

	def soft_update(self, target_model, source_model, tau):
		for target_param, param in zip(
			target_model.parameters(), source_model.parameters()
		):
			target_param.data.copy_(
				tau * param.data + (1.0 - tau) * target_param.data
			)
