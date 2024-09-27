class RequestOrder(object):
	def __init__(self, source, destination, current_time, human_only_order, value=1):
		self.origin_time = current_time
		self.pickup = source
		self.dropoff = destination
		self.human_only_order = human_only_order
		self.value = value
		self.id = -1
		self.deadline = -1

	def __str__(self):
		return(f'Order {self.id} ({self.pickup}, {self.dropoff}, {self.deadline})')

	def __repr__(self):
		return str(self)

	def update_state(self, current_time):
		self.state = [self.pickup, self.human_only_order, current_time]
		self.state_str = str(self.state)