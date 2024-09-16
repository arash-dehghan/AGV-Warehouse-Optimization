class Experience(object):
    envt = None
    def __init__(self, agents, request_ids, feasible_actions_all_agents, time, num_robots_charging, avg_robot_battery_percentage, avg_robot_capacity, avg_human_capacity, num_human_only_orders, num_both_orders, past_num_robots_charging, past_avg_robot_battery_percentage, past_avg_robot_capacity, past_avg_human_capacity, past_num_human_only_orders, past_num_both_orders):
        self.agents = agents
        self.request_ids = request_ids
        self.feasible_actions_all_agents = feasible_actions_all_agents
        self.time = time
        self.num_robots_charging = num_robots_charging
        self.avg_robot_battery_percentage = avg_robot_battery_percentage
        self.avg_robot_capacity = avg_robot_capacity
        self.avg_human_capacity = avg_human_capacity
        self.num_human_only_orders = num_human_only_orders
        self.num_both_orders = num_both_orders
        self.past_num_robots_charging = past_num_robots_charging
        self.past_avg_robot_battery_percentage = past_avg_robot_battery_percentage
        self.past_avg_robot_capacity = past_avg_robot_capacity
        self.past_avg_human_capacity = past_avg_human_capacity
        self.past_num_human_only_orders = past_num_human_only_orders
        self.past_num_both_orders = past_num_both_orders

        assert self.envt is not None
        assert len(agents) == (self.envt.num_humans + self.envt.num_robots)
        assert len(feasible_actions_all_agents) == (self.envt.num_humans + self.envt.num_robots)

        self.representation = {}
