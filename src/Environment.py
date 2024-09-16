from pandas import read_csv
from copy import deepcopy

class Environment():
	def __init__(self, num_humans, num_robots, epoch_length, horizon_length, road_speed_human, road_speed_robot, remove_vert, node_types, rebalancing_nodes, human_capacity, robot_capacity, battery_rate, charging_rate, delaytime):
		self.travel_time_human = read_csv(f'../data/data/human_{road_speed_human}_{remove_vert}_zone_traveltime.csv', header=None).values
		self.travel_time_robot = read_csv(f'../data/data/robot_{road_speed_robot}_{remove_vert}_zone_traveltime.csv', header=None).values
		self.shortest_path = read_csv(f'../data/data/zone_path_{remove_vert}.csv', header=None).values
		self.num_locations = len(self.travel_time_human)
		self.num_days_trained = 0
		self.num_humans = num_humans
		self.num_robots = num_robots
		self.num_agents = self.num_humans + self.num_robots
		self.human_capacity = human_capacity
		self.robot_capacity = robot_capacity
		self.start_epoch = 0
		self.stop_epoch = horizon_length
		self.epoch_length = epoch_length
		self.current_time = 0
		self.rebalancing_nodes = rebalancing_nodes
		# self.rebalancing_nodes = [27, 77, 22, 72]
		self.node_types = node_types
		self.battery_rate = battery_rate
		self.charging_rate = charging_rate
		self.car_capacity = max(human_capacity, robot_capacity)
		self.delaytime = delaytime

		# print(rebalancing_nodes)
		# exit()

	def get_travel_time(self, source, destination, is_human):
		return self.travel_time_human[source, destination] if is_human else self.travel_time_robot[source, destination]

	def get_next_location(self, source, destination):
		return self.shortest_path[source, destination]

	def get_nearest_charging_station(self, start):
		travel_times = {cs : self.get_travel_time(start, cs, False) for cs in self.node_types['C']}
		return min(travel_times, key=lambda k: travel_times[k])

	def calculate_battery_expenditure(self,next_location, time_to_next_location, target_location):
		# Calculate time until the next possible charging time
		time_until_next_charging = time_to_next_location + self.get_travel_time(next_location, target_location, False)
		return self.battery_rate * time_until_next_charging

	def get_rebalancing_next_location(self, next_location, time_to_next_location, rebalancing_location, is_human):
		if time_to_next_location >= self.epoch_length:
			return next_location, time_to_next_location - self.epoch_length
		else:
			time_remaining = self.epoch_length - time_to_next_location
			while time_remaining > 0:
				after_location = self.get_next_location(next_location, rebalancing_location)
				time_to_after_location = self.get_travel_time(next_location, after_location, is_human)
				if ((after_location == rebalancing_location) and (time_to_after_location == 0)):
					return rebalancing_location, 0.0
				if (time_to_after_location >= time_remaining):
					return after_location, time_to_after_location - time_remaining
				else:
					next_location = after_location
					time_remaining -= time_to_after_location
			print("ERROR!!!")
			exit()

	def simulate_rebalancing(self, agent, rebalancing_location, time):
		assert agent.capacity == 0
		assert len(agent.orders_to_pickup) + len(agent.orders_picked_up) == 0
		# Update agent location
		agent.next_location, agent.time_to_next_location = self.get_rebalancing_next_location(agent.next_location, agent.time_to_next_location, rebalancing_location, agent.is_human)
		# Update battery percentage if agent is a robot
		if not agent.is_human: agent.battery_percentage -= self.epoch_length * self.battery_rate
		assert agent.battery_percentage >= 0
		# Update agent state
		agent.update_state(time + self.epoch_length)

	def simulate_charging(self, agent, charging_station, time):
		assert agent.capacity == 0
		assert len(agent.orders_to_pickup) + len(agent.orders_picked_up) == 0
		assert not agent.is_human
		if agent.time_to_next_location + self.get_travel_time(agent.next_location, charging_station, agent.is_human) >= self.epoch_length:
			next_location, time_to_next_location = self.get_rebalancing_next_location(agent.next_location, agent.time_to_next_location, charging_station, agent.is_human)
			agent.next_location = next_location
			agent.time_to_next_location = time_to_next_location
			agent.battery_percentage -= self.epoch_length * self.battery_rate
		else:
			# Calculate time it will take robot to reach charging station and subtract that battery percentage
			time_to_charing_station = agent.time_to_next_location + self.get_travel_time(agent.next_location, charging_station, agent.is_human)
			agent.battery_percentage -= time_to_charing_station * self.battery_rate
			# Calculate the time that would be remaining to charge and add that amount of charging battery to the battery percentage
			time_left_to_charge = self.epoch_length - time_to_charing_station
			agent.battery_percentage += time_left_to_charge * self.charging_rate
			agent.battery_percentage = min(agent.battery_percentage, 100.0)
			# Update the next location and time to next location
			agent.next_location = charging_station
			agent.time_to_next_location = 0.0
		agent.update_state(time + self.epoch_length)

	def simulate_order_matching(self, agent, matching, time):
		if len(matching) != 0:
			agent.orders_to_pickup = deepcopy(matching)
			agent.capacity = len(agent.orders_picked_up) + len(agent.orders_to_pickup)
		orders_delivered, both_delivered, human_only_delivered = [], 0, 0
		time_remaining = self.epoch_length
		while True:
			if len(agent.orders_to_pickup) == 0:
				assert len(agent.orders_picked_up) > 0
				time_until_dropoff_area = agent.time_to_next_location + self.get_travel_time(agent.next_location, self.node_types['D'][0], agent.is_human)
				if time_until_dropoff_area > time_remaining:
					if time_remaining >= agent.time_to_next_location:
						time_remaining -= agent.time_to_next_location
						if not agent.is_human:
							agent.battery_percentage -= agent.time_to_next_location * self.battery_rate
						next_location = self.get_next_location(agent.next_location, self.node_types['D'][0])
						time_to_next_location = self.get_travel_time(agent.next_location, next_location, agent.is_human)
						agent.next_location = next_location
						agent.time_to_next_location = time_to_next_location
						if time_remaining == 0.0:
							agent.update_state(time + self.epoch_length)
							break
					else:
						agent.time_to_next_location -= time_remaining
						if not agent.is_human:
							agent.battery_percentage -= time_remaining * self.battery_rate
						agent.update_state(time + self.epoch_length)
						break
				else:
					assert agent.capacity == len(agent.orders_to_pickup) + len(agent.orders_picked_up)
					time_of_dropoff = time + (self.epoch_length - time_remaining) + time_until_dropoff_area
					orders_delivered = [time_of_dropoff - order.origin_time for order in agent.orders_picked_up]
					both_delivered = sum([1 for order in agent.orders_picked_up if not order.human_only_order])
					human_only_delivered = len(agent.orders_picked_up) - both_delivered
					assert len(orders_delivered) == both_delivered + human_only_delivered
					for o in orders_delivered:
						if o < 0:
							print("DAMN")
							exit()
					agent.next_location, agent.time_to_next_location = self.node_types['D'][0], 0.0
					agent.capacity, agent.orders_to_pickup, agent.orders_picked_up = 0, [], []
					if not agent.is_human:
						agent.battery_percentage -= time_remaining * self.battery_rate
					agent.update_state(time + self.epoch_length)
					break
			else:
				if agent.time_to_next_location > time_remaining:
					agent.time_to_next_location -= time_remaining
					if not agent.is_human:
						agent.battery_percentage -= time_remaining * self.battery_rate
					agent.update_state(time + self.epoch_length)
					break
				else:
					time_remaining -= agent.time_to_next_location
					if not agent.is_human:
						agent.battery_percentage -= agent.time_to_next_location * self.battery_rate
					agent.time_to_next_location = 0.0
					agent.orders_picked_up.extend(order for order in agent.orders_to_pickup if order.pickup == agent.next_location)
					agent.orders_to_pickup = [order for order in agent.orders_to_pickup if order.pickup != agent.next_location]
					if len(agent.orders_to_pickup) > 0: 
						next_location_to_visit = self.get_next_location(agent.next_location, agent.orders_to_pickup[0].pickup)
						next_location_to_visit_time = self.get_travel_time(agent.next_location, next_location_to_visit, agent.is_human)
						agent.next_location = next_location_to_visit
						agent.time_to_next_location = next_location_to_visit_time
		return orders_delivered, both_delivered, human_only_delivered


					