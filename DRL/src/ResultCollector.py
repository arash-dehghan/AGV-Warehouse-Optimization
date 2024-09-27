import numpy as np
import matplotlib.pyplot as plt

class ResultCollector(object):
	def __init__(self):
		stats_names = ['Orders Seen', 'Number of Human-only Orders', 'Number of Both Orders', 'Orders Served', 'Orders Served by Humans', 'Orders Served by Robots', 'Average # of Feasible Order Matchings Human', 'Average # of Feasible Order Matchings Robot', 'Number of Robots Charging', 'Average Robot Battery Percentage', 'Average Assigned # of Orders Robot', 'Average Assigned # of Orders Human', 'Average Matching Sizes', 'Average Delivery Time']
		self.results = {stat : {} for stat in stats_names}
		self.std_served_stats = {}
		self.std_seen_stats = {}

	def update_results(self, iteration, results):
		stats_names = ['Orders Seen', 'Number of Human-only Orders', 'Number of Both Orders', 'Orders Served', 'Orders Served by Humans', 'Orders Served by Robots', 'Average # of Feasible Order Matchings Human', 'Average # of Feasible Order Matchings Robot', 'Number of Robots Charging', 'Average Robot Battery Percentage', 'Average Assigned # of Orders Robot', 'Average Assigned # of Orders Human', 'Average Matching Sizes', 'Average Delivery Time']
		for i, stat in enumerate(stats_names):
			overall = np.nanmean([days_result[i] for days_result in results], axis=0)
			self.results[stat][iteration] = overall