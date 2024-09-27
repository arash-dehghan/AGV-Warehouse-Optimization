class Experience(object):
    envt = None
    def __init__(self, previous_states, rewards, next_states, available_agent_actions):
        self.previous_states = previous_states
        self.rewards = rewards
        self.next_states = next_states
        self.available_agent_actions = available_agent_actions
        # self.is_end = is_end
