import sys
sys.dont_write_bytecode = True
import networkx as nx
import matplotlib.pyplot as plt
import numpy
import argparse
import pandas as pd
from DataGenerator import DataGenerator
import os
import pickle
import copy

def create_travel_files(G, H):
	shortest_paths = dict(nx.all_pairs_shortest_path(G)) # Create shortest path and length of shortest path dictionaries
	create_next_location_file(G.nodes, shortest_paths) # Create next location file
	create_time_to_location_file(G, shortest_paths, 'robot') # Create travel time file for robots
	create_time_to_location_file(H, shortest_paths, 'human') # Create travel time file for humans

def create_next_location_file(nodes, sp):
	overall_next_nodes = []
	for start_node in nodes:
		next_node = []
		for dest_node in nodes:
			path = sp[start_node][dest_node]
			if len(path) == 1:
				next_node.append(path[0])
			else:
				next_node.append(path[1])
		overall_next_nodes.append(next_node)
	next_node_df = pd.DataFrame(overall_next_nodes)
	next_node_df.to_csv(f'data/zone_path_{args.remove_vert}.csv', header=False, index=False)

def create_time_to_location_file(G, sp, name):
	overall_times = []
	for start_node in G.nodes:
		node_distances = []
		for end_node in G.nodes:
			path = sp[start_node][end_node]
			if len(path) == 1:
				node_distances.append(0.0)
			else:
				time = sum([G[path[index]][path[index+1]]['weight'] for index in range(len(path)-1)])
				node_distances.append(time)
		overall_times.append(node_distances)
	time_df = pd.DataFrame(overall_times)
	time_df.to_csv(f'data/{name}_{args.edge_travel_time_human if name == "human" else args.edge_travel_time_robot}_{args.remove_vert}_zone_traveltime.csv', header=False, index=False)

def create_grid_with_charging_stations():
	# Assert existence of 1-3 charging stations
	if (args.num_cs > 3) or (args.num_cs < 1): raise ValueError("Number of charging stations must be between 1 and 3.")

	# Create a grid graph
	G = nx.grid_2d_graph(args.num_rows, args.num_cols)

	# Define corner nodes
	corners = [(0, 0), (0, args.num_cols-1), (args.num_rows-1, 0), (args.num_rows-1, args.num_cols-1)]

	# Define drop-off area node and charging station nodes
	dropoff_area, charging_stations = (0,0), corners[1:args.num_cs + 1]
	
	# Define all node types in the warehouse
	for node in G.nodes():
		if node == dropoff_area:
			G.nodes[node]['type'] = 'D' # If node is the drop-off area node, set to 'D' for 'Drop-off Area'
		elif node in charging_stations:
			G.nodes[node]['type'] = 'C' # If node is a charging station node, set to 'C' for 'Charging Station'
		elif (node[0] == 0) or (node[0] == args.num_rows - 1):
			G.nodes[node]['type'] = 'P' # If node is on the top or bottom row but not a drop-off area or charging station, set to 'P' for pathway
		else:
			G.nodes[node]['type'] = 'S' # Otherwise, node is a shelf node, set to 'S' for shelf
	
	# Remove horizontal edges from non-outer shelf nodes
	if args.remove_vert:
		for node in G.nodes():
			if (G.nodes[node]['type'] == 'S') and (node[0] != 0) and (node[0] != args.num_rows - 1):
				right_node, left_node = (node[0], node[1]-1), (node[0], node[1]+1)
				if (right_node in G) and (G.has_edge(node, right_node)): G.remove_edge(node, right_node)
				if (left_node in G) and (G.has_edge(node, left_node)): G.remove_edge(node, left_node)

	# Set the coordinates of the node to 'coordinate' and assign unique IDs to all nodes
	for node in G: G.nodes[node]['coordinate'] = node
	G = nx.relabel_nodes(G, {node : i for i, node in enumerate(G.nodes)})

	# Create duplicate of graph
	H = copy.deepcopy(G)

	# Set edge weight of each edge to the 'edge_travel_time' for each of the graphs. G is robot, H is human
	for start,end in G.edges: G[start][end]['weight'] = args.edge_travel_time_robot
	for start,end in H.edges: H[start][end]['weight'] = args.edge_travel_time_human

	# Iterate over all 'S' nodes and check their neighbors
	s_nodes_connected_to_c_or_p = []
	for node in G.nodes():
		if G.nodes[node]['type'] == 'S':
			neighbors = G.neighbors(node)
			for neighbor in neighbors:
				neighbor_type = G.nodes[neighbor]['type']
				if neighbor_type in ['C', 'P', 'D']:
					s_nodes_connected_to_c_or_p.append(node)
					break  # We found a connection, no need to check further neighbors

	return G, H, s_nodes_connected_to_c_or_p

def visualize_graph(G):
	pos_coordinates = {node: G.nodes[node]['coordinate'] for node in G.nodes()}
	node_colors = []
	labels = {}
	ids = {}
	for node in G.nodes():
		node_type = G.nodes[node]['type']
		if node_type == 'D':
			node_colors.append('green')
			labels[node] = 'D'
			ids[node] = node
		elif node_type == 'C':
			node_colors.append('red')
			labels[node] = 'C'
			ids[node] = node
		elif node_type == 'P':
			node_colors.append('blue')
			labels[node] = 'P'
			ids[node] = node
		else:
			node_colors.append('yellow')
			labels[node] = 'S'
			ids[node] = node
	
	nx.draw(G, pos=pos_coordinates, node_color=node_colors, node_size=500, labels=ids)
	plt.show()

def get_poisson_order_distributions(G):
	# Define node IDs for each node type
	node_types = {t: [node for node in G.nodes() if G.nodes[node]['type'] == t] for t in ['D', 'C', 'P', 'S']}
	
	# Generate probabilities using Poisson distribution
	start_probabilities = {t : np.poisson(args.lam, len(node_types['S'])) for t in range(int(args.horizon_length / args.epoch_length))}

	# # Normalize the probabilities
	normalized_start_probabilities = {t : probability / numpy.sum(probability) for t, probability in start_probabilities.items()}
	
	return node_types, normalized_start_probabilities

def create_order_distribution_means(n = 10000):
	# To generate a left-skewed distribution, we can directly use the beta distribution without taking the reciprocal
	X = int(args.horizon_length / args.epoch_length)

	# Generate left-skewed data using beta distribution
	left_skewed_data = np.beta(5, 2, n)

	# Generate histogram data for left-skewed distribution
	hist_left, bin_edges_left = numpy.histogram(left_skewed_data, bins=1000, density=True)

	# Find cumulative sum of histogram data for left-skewed distribution
	cum_sum_left = numpy.cumsum(hist_left)
	cum_sum_left = cum_sum_left / cum_sum_left[-1]  # Normalize to 1

	# Find X equidistant points on the cumulative distribution for left-skewed distribution
	indices_left = numpy.linspace(0, cum_sum_left[-1], X+2)[1:-1]  # Exclude 0 and 1
	equidistant_points_left = [bin_edges_left[numpy.searchsorted(cum_sum_left, i)] for i in indices_left]

	# Plotting
	# plt.figure(figsize=(10, 6))
	# plt.hist(left_skewed_data, bins=1000, density=True, alpha=0.6, label="Left-skewed Distribution")
	# plt.scatter(equidistant_points_left, [0]*len(equidistant_points_left), color='red', s=50, label='Equidistant Points')
	# plt.legend()
	# plt.title("Left-skewed Distribution with Equidistant Points")
	# plt.show()

	# Get the y-values (densities) for the equidistant points for left-skewed distribution
	y = numpy.array([hist_left[numpy.searchsorted(bin_edges_left[:-1], point)] for point in equidistant_points_left]) * args.order_multiplier

	# Create an area of one standard deviation around the line, ensuring it is >= 0
	y_upper = numpy.clip(y + 1, 0, None)  # Upper bound
	y_lower = numpy.clip(y - 1, 0, None)  # Lower bound

	# Plotting
	plt.figure(figsize=(8, 6))
	x = range(0,1440,5)
	plt.plot(x,y)
	plt.xlabel('Time of Day (in Minutes)', fontsize = 15)
	plt.ylabel('Avg. Number of Incoming Orders', fontsize = 15)
	plt.fill_between(x, y_lower, y_upper, color='red', alpha=0.2, label='Std Dev')
	plt.gca().set_facecolor('#f2f2f2')
	plt.grid(True, linestyle='--', color='grey', alpha=0.5)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.tight_layout(pad=3.0)
	# plt.legend(fontsize=15)
	plt.show()
	exit()

	return numpy.array([hist_left[numpy.searchsorted(bin_edges_left[:-1], point)] for point in equidistant_points_left]) * args.order_multiplier

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_rows', '--num_rows', type=int, default=10)
	parser.add_argument('-num_cols', '--num_cols', type=int, default=10)
	parser.add_argument('-num_cs', '--num_cs', type=int, default=2)
	parser.add_argument('-remove_vert', '--remove_vert', type=int, default=1)
	parser.add_argument('-lam', '--lam', type=float, default=1.0)
	parser.add_argument('-horizon_length', '--horizon_length', type=int, default=1440)
	parser.add_argument('-edge_travel_time_human', '--edge_travel_time_human', type=float, default=1)
	parser.add_argument('-edge_travel_time_robot', '--edge_travel_time_robot', type=float, default=1)
	parser.add_argument('-epoch_length', '--epoch_length', type=int, default=5)
	parser.add_argument('-order_multiplier', '--order_multiplier', type=int, default=5)
	parser.add_argument('-percentage_only_human_orders', '--percentage_only_human_orders', type=float, default=0.0)
	parser.add_argument('-seed', '--seed', type=int , default=1)
	args = parser.parse_args()
	np = numpy.random.RandomState(args.seed)

	filename = f'{args.num_rows * args.num_cols}_{args.num_cs}_{args.remove_vert}_{args.horizon_length}_{args.edge_travel_time_human}_{args.edge_travel_time_robot}_{args.epoch_length}_{args.order_multiplier}_{args.percentage_only_human_orders}_{args.seed}'

	# Create network graphs for humans and robots
	G, H, rebalancing_nodes = create_grid_with_charging_stations()
	visualize_graph(G)

	# Create location and delay files
	create_travel_files(G,H)

	# Get poison distribution for order arrivals in nodes
	node_types, normalized_start_probabilities  = get_poisson_order_distributions(G)

	# Generate order arrivals distribution means
	distribution_means = create_order_distribution_means()

	Data = DataGenerator(G = G,
		H = H,
		node_types = node_types,
		start_probs = normalized_start_probabilities,
		dist_means = distribution_means,
		p = args.percentage_only_human_orders,
		rebalancing_nodes = rebalancing_nodes,
		seed = args.seed)

	if not os.path.exists(f'generations/{filename}'):
		os.makedirs(f'generations/{filename}')

	with open(f'generations/{filename}/data_{filename}.pickle', 'wb') as handle:
		pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)



