from RequestOrder import RequestOrder
import numpy

class DataGenerator(object):
	def __init__(self, G, H, node_types, start_probs, dist_means, p, rebalancing_nodes, seed):
		self.G = G
		self.H = H
		self.node_types = node_types
		self.start_probs = start_probs
		self.dist_means = dist_means
		self.p = p
		self.rebalancing_nodes = rebalancing_nodes
		self.seed = seed
		self.np = numpy.random.RandomState(seed)

	def get_requests(self, time):
		number_of_requests = self.get_number_requests(time)
		locations = self.get_locations(number_of_requests, time)
		order_types = self.get_order_types(number_of_requests)
		return self.create_requests(time,locations, order_types)

	def get_number_requests(self,time):
		avg = self.dist_means[time]
		num_requests = -1
		while num_requests < 0:
			num_requests = int(self.np.normal(loc=avg, scale=1))
		return num_requests

	def get_locations(self, n, time):
		return [self.get_start_location(time) for request in range(n)]

	def get_start_location(self, time):
		return self.np.choice(self.node_types['S'], p = self.start_probs[time])

	def get_order_types(self, number_of_requests):
		return self.np.binomial(1, self.p, number_of_requests).astype(bool)

	def create_requests(self, time, locations, order_types):
		return [RequestOrder(locations[i], self.node_types['D'][0], time, order_types[i]) for i in range(len(locations))]

	def create_test_scenarios(self,num_days):
		test_scenarios = {day: {} for day in range(num_days)}
		for day in range(num_days):
			for time in self.start_probs.keys():
				test_scenarios[day][time] = self.get_requests(time)
		return test_scenarios
			
	def get_start_locations(self, num_agents):
		return self.np.choice(self.node_types['S'], size=num_agents, replace=True)

	def get_battery_percentage(self, start_locations, battery_rate, travel_times):
		battery_percentages = []
		for location in start_locations:
			closest_cs_time = min([travel_times[location][charging_station] for charging_station in self.node_types['C']])
			battery_needed = battery_rate * closest_cs_time
			assert battery_needed <= 100
			battery = min(100, self.np.choice(range(int(numpy.ceil(battery_needed)),101)))
			battery_percentages.append(battery)
		return battery_percentages


