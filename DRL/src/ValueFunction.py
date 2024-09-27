import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import numpy as np
from copy import deepcopy
from ReplayBuffer import SimpleReplayBuffer, PrioritizedReplayBuffer
import math

class DRL(nn.Module):
	def __init__(self, envt):
		super(DRL, self).__init__()
		self.envt = envt

		# Embedding for location
		self.location_embed = nn.Embedding(num_embeddings=self.envt.num_locations + 1, embedding_dim=100)

		# LSTM for path embedding
		self.lstm = nn.LSTM(input_size=101, hidden_size=200, batch_first=True)

		# Dense layers for embeddings
		self.time_embedding = nn.Linear(1, 100)

		# State embeddings and final dense layers
		self.state_embed_1 = nn.Linear(312, 300)
		self.state_embed_2 = nn.Linear(300, 300)
		self.output_layer = nn.Linear(300, 1)

	def forward(self, combined_input_stack):
		"""
		Expects a stacked input tensor with the following structure:
		combined_input_stack.shape = (stack_size, batch_size, input_size)
		"""
		outputs = []
		
		# Iterate over the stack of inputs
		for combined_input in combined_input_stack:
			# Process each combined input individually

			# Calculate sizes for slicing
			capacity = self.envt.car_capacity
			seq_len = capacity * 2 + 1  # Sequence length for locations and delays

			# First section: location inputs
			locations_input = combined_input[:, :seq_len].long()  # Shape: (batch_size, seq_len)
			path_location_embed = self.location_embed(locations_input)  # Shape: (batch_size, seq_len, embedding_dim)

			# Next section: delays
			delay_start = seq_len
			delay_end = delay_start + seq_len
			delays_input = combined_input[:, delay_start:delay_end].unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)

			# Mask delays and concatenate with location embeddings
			delay_masked = torch.where(delays_input == -1, torch.zeros_like(delays_input), delays_input)
			path_input = torch.cat((path_location_embed, delay_masked), dim=-1)  # Shape: (batch_size, seq_len, embedding_dim + 1)

			# LSTM for path embeddings
			path_embed, _ = self.lstm(path_input)  # Output shape: (batch_size, seq_len, hidden_size)
			path_embed = path_embed[:, -1, :]  # Get the last hidden state: (batch_size, hidden_size)

			# Rest of the inputs (current time, capacity, etc.)
			other_inputs = combined_input[:, delay_end:]  # Shape: (batch_size, num_other_inputs)

			# Extract individual inputs
			current_time_input = other_inputs[:, :1]  # Shape: (batch_size, 1)
			capacity_input = other_inputs[:, 1:2]     # Shape: (batch_size, 1)
			is_human_input = other_inputs[:, 2:3]     # Shape: (batch_size, 1)
			battery_percentage_input = other_inputs[:, 3:4]
			num_robots_charging_input = other_inputs[:, 4:5]
			avg_robot_battery_percentage_input = other_inputs[:, 5:6]
			avg_robot_capacity_input = other_inputs[:, 6:7]
			avg_human_capacity_input = other_inputs[:, 7:8]
			num_human_only_orders_input = other_inputs[:, 8:9]
			num_both_orders_input = other_inputs[:, 9:10]
			action_input = other_inputs[:, 10:]  # Now captures the entire action vector (size 3)

			# Embedding for current time
			current_time_embed = F.elu(self.time_embedding(current_time_input))  # Shape: (batch_size, 100)

			# Concatenate all inputs for the state embedding
			state_embed = torch.cat([
				path_embed,                         # Shape: (batch_size, 200)
				current_time_embed,                 # Shape: (batch_size, 100)
				capacity_input,                     # Shape: (batch_size, 1)
				is_human_input,
				battery_percentage_input,
				num_robots_charging_input,
				avg_robot_battery_percentage_input,
				avg_robot_capacity_input,
				avg_human_capacity_input,
				num_human_only_orders_input,
				num_both_orders_input,
				action_input                        # Shape: (batch_size, 3)
			], dim=-1)  # Expected total shape: (batch_size, 312)

			# Pass through dense layers
			state_embed = F.elu(self.state_embed_1(state_embed))
			state_embed = F.elu(self.state_embed_2(state_embed))

			# Output predicted value
			output = self.output_layer(state_embed)
			outputs.append(output)

		# Stack outputs back into a tensor
		return torch.stack(outputs, dim=0)  # Shape: (stack_size, batch_size, 1)


class NeurADP():
	def __init__(self, envt, load_model_loc=''):
		# General information
		self.envt = envt
		self.total_train_days = 60

		self.actions_to_one = {'M': (1, 0, 0), 'C': (0, 1, 0), 'R': (0, 0, 1)}
		self.one_to_actions = {(1, 0, 0): 'M', (0, 1, 0): 'C', (0, 0, 1): 'R'}

		# Matching NN information
		self.model = DRL(self.envt)
		self.target = DRL(self.envt)
		self.target.load_state_dict(self.model.state_dict())
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.99, patience=500, verbose=False)
		self.gamma = 0.9
		self.target_update_tau = 0.001

		# Matching Replay Buffer information
		min_len_replay_buffer = 5000
		self.num_min_train_samples = 1000
		self.num_samples = 10
		epochs_in_episode = (self.envt.stop_epoch - self.envt.start_epoch) / self.envt.epoch_length
		len_replay_buffer = max((min_len_replay_buffer, epochs_in_episode))
		self.replay_buffer = PrioritizedReplayBuffer(MAX_LEN=int(len_replay_buffer))

	def get_action_values(self, is_training, agents, time, num_robots_charging, avg_robot_battery_percentage, avg_robot_capacity, avg_human_capacity, num_human_only_orders, num_both_orders):
		final_acts = []
		NN_input_states = []
		
		# Iterate over agents
		for agent in agents:
			acts = []
			agent_NN_inputs = {}

			# Prepare a stack of inputs for all actions
			input_stack = []
			for act_name, action in self.actions_to_one.items():
				info = self._format_input(agent, time, num_robots_charging, avg_robot_battery_percentage, avg_robot_capacity, avg_human_capacity, num_human_only_orders, num_both_orders, action)
				inp = self._NN_format(info)
				agent_NN_inputs[act_name] = inp
				input_stack.append(inp)

			# Convert the input stack into a tensor and pass through the model
			input_stack_tensor = torch.cat(input_stack, dim=0).unsqueeze(0)  # Shape: (1, len(actions), input_dim)
			outputs = self.model(input_stack_tensor)  # Shape: (1, len(actions), 1)

			# Extract the outputs and sort them
			for i, act_name in enumerate(self.actions_to_one.keys()):
				output = outputs[0, i, 0].item()
				acts.append((act_name, output))

			acts = sorted(acts, key=lambda x: x[1], reverse=True)

			

			if is_training:
				n = 1 / (self.envt.num_days_trained + 1)
				do_random = np.random.choice([True, False], p=[n, 1 - n])
				if do_random:
					np.random.shuffle(acts)

			acts = [a[0] for a in acts]
			final_acts.append(acts)
			NN_input_states.append(agent_NN_inputs)

		return final_acts, NN_input_states


	def _format_input(self, agent, current_time, num_robots_charging, avg_robot_battery_percentage,
					  avg_robot_capacity, avg_human_capacity, num_human_only_orders, num_both_orders, action_input):
		# Normalizing Inputs
		locations_input = self._add_locations(agent.next_location, agent.orders_to_pickup)
		delays_input = self._add_delays(agent.time_to_next_location, agent.orders_to_pickup, agent.orders_picked_up, current_time)
		current_time_input = (current_time - self.envt.start_epoch) / (self.envt.stop_epoch - self.envt.start_epoch)
		capacity_input = agent.capacity / self.envt.car_capacity
		is_human_input = agent.is_human
		battery_percentage_input = agent.battery_percentage / 100

		# Auxiliary information
		num_robots_charging_input = 0 if self.envt.num_robots == 0 else num_robots_charging / self.envt.num_robots
		avg_robot_battery_percentage_input = avg_robot_battery_percentage
		avg_robot_capacity_input = avg_robot_capacity
		avg_human_capacity_input = avg_human_capacity
		num_human_only_orders_input = 0.0 if num_both_orders == 0 else num_human_only_orders / num_both_orders
		num_both_orders_input = num_both_orders

		# num_robots_charging_input = 0
		# avg_robot_battery_percentage_input = 0
		# avg_robot_capacity_input = 0
		# avg_human_capacity_input = 0
		# num_human_only_orders_input = 0
		# num_both_orders_input = 0


		# Convert action_input to numpy array if it's a list
		action_input = np.array(action_input, dtype=np.float32)

		# Return a list of inputs
		return [locations_input, delays_input, current_time_input, capacity_input, is_human_input, battery_percentage_input,
				num_robots_charging_input, avg_robot_battery_percentage_input, avg_robot_capacity_input,
				avg_human_capacity_input, num_human_only_orders_input, num_both_orders_input, action_input]

	def _NN_format(self, info):
		# Extract components
		locations_input = info[0]  # numpy array of ints, shape: (seq_len,)
		delays_input = info[1].flatten()  # numpy array, shape: (seq_len,)
		other_inputs = np.array(info[2:-1], dtype=np.float32)  # array of scalars, shape: (10,)
		action_input = np.array(info[-1], dtype=np.float32)  # action vector, shape: (3,)

		# Convert locations_input to float32 for concatenation
		locations_input_float = locations_input.astype(np.float32)

		# Concatenate all inputs
		combined_input_np = np.concatenate((locations_input_float, delays_input, other_inputs, action_input), axis=0)

		# Convert to PyTorch tensor and add batch dimension
		combined_input = torch.tensor(combined_input_np, dtype=torch.float32).unsqueeze(0)  # Shape: (1, total_input_size)

		return combined_input

	def _add_locations(self, next_location, orders_to_pickup):
		locations_input = np.zeros(shape=(self.envt.car_capacity * 2 + 1,), dtype='int32')
		locations_input[self.envt.car_capacity] = next_location + 1
		for i, order in enumerate(orders_to_pickup):
			j = self.envt.car_capacity + i + 1
			locations_input[j] = order.pickup + 1
		return locations_input

	def _add_delays(self, time_to_next_location, orders_to_pickup, orders_picked_up, current_time):
		delays = np.zeros(shape=(self.envt.car_capacity * 2 + 1, 1)) - 1
		delays[self.envt.car_capacity] = time_to_next_location / self.envt.delaytime
		for i, order in enumerate(orders_picked_up):
			delays[i] = (order.deadline - current_time) / self.envt.delaytime
		for i, order in enumerate(orders_to_pickup):
			j = i + self.envt.car_capacity + 1
			delays[j] = (order.deadline - current_time) / self.envt.delaytime
		return delays

	def make_matchings(self, agents, feasible_actions, action_values, humans_first, current_orders):
		human_ids, robot_ids = [a.id for a in agents if a.is_human], [a.id for a in agents if not a.is_human]
		agent_ids = human_ids + robot_ids if humans_first else robot_ids + human_ids
		assigned_actions = {}
		labeled_assignments = []
		orders_dealt_with = []
		available_agent_actions = []
		orders_served = 0
		for agent_id in agent_ids:
			agent = agents[agent_id]
			agent_feasible_actions = feasible_actions[agent_id]
			matching_actions, rebalancing_actions, charging_actions = self.breakdown_actions(agent_feasible_actions)
			matching_actions = self.filter_matching_actions(matching_actions, orders_dealt_with)

			action_keys = {'M' : matching_actions, 'R' : rebalancing_actions, 'C': charging_actions}

			agent_available_actions = [label for label, acts in action_keys.items() if len(acts) > 0]
			available_agent_actions.append(agent_available_actions)

			action_to_do = self.get_doable_action(action_keys, action_values[agent_id])
			labeled_assignments.append(action_to_do)

			if action_to_do == 'R':
				assert len(rebalancing_actions)
				r_dict = {i: r for i,r in enumerate(rebalancing_actions)}
				loc = np.random.choice(list(r_dict.keys()))
				assigned_actions[agent.id] = r_dict[loc]
			elif action_to_do == 'M':
				assert len(matching_actions)
				assigned_actions[agent.id] = matching_actions[0]
				if matching_actions[0][1] != 'O_':
					served = self.get_new_orders(matching_actions[0][1])
					for o in served:
						orders_dealt_with.append(o)
						orders_served += 1
			elif action_to_do == 'C':
				assert len(charging_actions)
				assigned_actions[agent.id] = charging_actions[0]

		assert len(assigned_actions) == len(agents)
		assert orders_served <= len(current_orders)

		return assigned_actions, labeled_assignments, available_agent_actions

	def breakdown_actions(self, actions):
		matching_actions = [act for act in actions if act[1][0] == 'O']
		rebalancing_actions = [act for act in actions if act[1][0] == 'R']
		charging_actions = [act for act in actions if act[1][0] == 'C']
		return matching_actions, rebalancing_actions, charging_actions

	def filter_matching_actions(self, actions, orders_dealt_with):
		matchable_actions = []
		for action in actions:
			matchable = True
			newly_matched_orders = self.get_new_orders(action[1])
			for order_id in newly_matched_orders:
				if order_id in orders_dealt_with:
					matchable = False
					break
			if matchable:
				matchable_actions.append(action)
		matchable_actions = sorted(matchable_actions, key=self.custom_sort_key)
		return matchable_actions

	def get_new_orders(self, label):
		return [l for l in label.split('_')[1:] if l != '']

	def custom_sort_key(self, item):
		served = len(self.get_new_orders(item[1]))
		return (-served, item[2])

	def get_doable_action(self, action_keys, action_values):
		for act in action_values:
			if len(action_keys[act]) > 0:
				return act

		print("ERROR!!")
		exit()

	def remember(self, experience):
		self.replay_buffer.add(experience)

	def update(self):
		if self.num_min_train_samples > len(self.replay_buffer):
			return

		self.model.train()
		self.target.train()

		beta = min(1, 0.4 + 0.6 * (self.envt.num_days_trained / self.total_train_days))
		experiences, weights, batch_idxes = self.replay_buffer.sample(self.num_samples, beta)
		weights = torch.tensor(weights, dtype=torch.float32)

		# experiences = self.pricing_replay_buffer.sample(self.pricing_num_samples)
		total_loss = 0.0

		for experience_idx, (experience, batch_idx) in enumerate(zip(experiences, batch_idxes)):
			experience = deepcopy(experience)

			direct_rewards = np.array(experience.rewards)
			future_Q_values = self.get_future_Q(experience.next_states, experience.available_agent_actions)
			scores = direct_rewards + self.gamma * future_Q_values
			supervised_targets = torch.tensor(np.array(scores).reshape((-1, 1)), dtype=torch.float32)

			current_inputs = torch.stack(experience.previous_states)
			outputs = self.model(current_inputs).squeeze(-1)

			loss_function = torch.nn.MSELoss(reduction='none')
			loss = loss_function(outputs, supervised_targets)
			weighted_loss = (loss * weights[experience_idx]).mean()
			total_loss += float(weighted_loss)

			# Backward pass and optimization
			self.optimizer.zero_grad()
			weighted_loss.backward()
			self.optimizer.step()

			predicted_values = self.model(current_inputs)
			loss = float(torch.mean((predicted_values - supervised_targets) ** 2 + 1e-6))
			self.replay_buffer.update_priorities([batch_idx], [loss])

		self.soft_update()

		avg_loss = total_loss / self.num_samples
		self.scheduler.step(avg_loss)

		self.model.eval()
		self.target.eval()

		return avg_loss

	def get_future_Q(self, future_states, available_actions):
		values = []
		for agent_states, available_acts in zip(future_states, available_actions):
			a = {}
			for label, state in agent_states.items():
				if label in available_acts:
					a[label] = self.target(state.unsqueeze(0)).item()
			best_act = max(a, key=a.get)
			best_val = a[best_act]

			values.append(best_val)

		return np.array(values)

	def soft_update(self):
		with torch.no_grad():
			for target_param, source_param in zip(self.target.parameters(), self.model.parameters()):
				updated_weight = self.target_update_tau * source_param.data + (1 - self.target_update_tau) * target_param.data
				target_param.data.copy_(updated_weight)
		



