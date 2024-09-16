import numpy as np
# np.random.seed(0)

class LearningAgent(object):
	def __init__(self, agent_id, is_human, starting_location, battery_percentage, current_time = 0):
		self.id = agent_id
		self.is_human = is_human
		self.next_location = starting_location
		self.time_to_next_location = 0.0
		self.capacity = 0
		self.battery_percentage = battery_percentage
		self.orders_picked_up = []
		self.orders_to_pickup = []
		self.update_state(current_time)

	def __str__(self):
		return(f'Agent {self.id} [{self.is_human}, {self.battery_percentage}]')

	def __repr__(self):
		return str(self)

	def update_state(self,current_time):
		self.state = [self.is_human, self.next_location, self.time_to_next_location, self.capacity, self.battery_percentage, self.orders_picked_up, self.orders_to_pickup, current_time]
		self.state_str = str(self.state)