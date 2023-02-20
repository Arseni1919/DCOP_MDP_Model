import random

from environments.nodes_from_pic import build_graph_nodes
from plot_fucntions.plot_functions import *
from globals import *


class EnvAgent:
    def __init__(self, start_pos, goal_pos, n_agent):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.n_agent = n_agent
        self.name = f'agent_{n_agent}'
        self.pos = self.start_pos
        self.actions = [0, 1, 2, 3, 4]  # 1 - up, 2 - right, 3 - down, 4 - left, 0 - stay
        self.terminated = False
        self.truncated = False
        self.reward = 0


def create_env_agents(start_positions, goal_positions):
    agents = []
    agents_dict = {}
    for n_agent, (start_pos, goal_pos) in enumerate(zip(start_positions, goal_positions)):
        agent = EnvAgent(start_pos, goal_pos, n_agent)
        agents.append(agent)
        agents_dict[agent.name] = agent
    return agents, agents_dict


class SquareEnv:
    def __init__(self, start_positions: list, goal_positions: list, nodes: list, nodes_dict: dict, height: int, width: int):
        self.start_positions = start_positions
        self.goal_positions = goal_positions
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.height = height
        self.width = width

        self.env_agents = None
        self.env_agents_dict = None
        self.n_agents = None
        self.step_count = None
        self.max_steps = 100
        self.reward_kinds = {'col': -100, 'goal': 100, 'step': -1}

        self.fig, self.ax = plt.subplots(2, 1)

    def build_obs(self, pos):
        # field = self.get_states()
        # field[f'{pos[0]}_{pos[1]}'] = 1
        pos_name = f'{pos[0]}_{pos[1]}'
        return pos_name

    def get_states(self):
        # return np.zeros((self.width, self.height))
        return {node.xy_name: 0 for node in self.nodes}

    def reset(self):
        self.env_agents, self.env_agents_dict = create_env_agents(self.start_positions, self.goal_positions)
        self.n_agents = len(self.env_agents)
        self.step_count = 0
        observations, rewards, terminated, truncated, info = {}, {}, {}, {}, {}
        for agent in self.env_agents:
            observations[agent.name] = self.build_obs(agent.pos)
            rewards[agent.name] = -1
            terminated[agent.name] = False
            truncated[agent.name] = False

        return observations, rewards, terminated, truncated, info

    def get_agent(self, agent_name):
        return self.env_agents_dict[agent_name]

    def sample_action(self, agent_name):
        agent = self.get_agent(agent_name)
        return random.choice(agent.actions)

    def sample_actions(self):
        return {agent.name: random.choice(agent.actions) for agent in self.env_agents}

    def get_next_possible_nodes(self, state_name, action):
        # 1 - up, 2 - right, 3 - down, 4 - left, 0 - stay
        state_node = self.nodes_dict[state_name]
        straight = state_node
        to_the_right = state_node
        to_the_left = state_node
        if action == 1:  # up
            straight = state_node.up_node
            to_the_right = state_node.right_node
            to_the_left = state_node.left_node
        elif action == 2:  # right
            straight = state_node.right_node
            to_the_right = state_node.down_node
            to_the_left = state_node.up_node
        elif action == 3:  # down
            straight = state_node.down_node
            to_the_right = state_node.left_node
            to_the_left = state_node.right_node
        elif action == 4:  # left
            straight = state_node.left_node
            to_the_right = state_node.up_node
            to_the_left = state_node.down_node

        dynamics_dict = {}
        # truncated_dict = {}
        dynamics_dict[(state_node.xy_name, True)] = 0
        if straight is not None:
            dynamics_dict[(straight.xy_name, False)] = 0.8
        else:
            dynamics_dict[(state_node.xy_name, True)] += 0.8
            # truncated_dict[]
        if to_the_right is not None:
            dynamics_dict[(to_the_right.xy_name, False)] = 0.1
        else:
            dynamics_dict[(state_node.xy_name, True)] += 0.1
        if to_the_left is not None:
            dynamics_dict[(to_the_left.xy_name, False)] = 0.1
        else:
            dynamics_dict[(state_node.xy_name, True)] += 0.1

        return dynamics_dict

    def get_next_possible_rewards(self, agent_name, dynamics_dict):
        reward_dict = {}
        termination_dict = {}
        agent = self.env_agents_dict[agent_name]
        for (node_name, truncated), prob in dynamics_dict.items():
            reward_dict[(node_name, truncated)] = 0
            termination_dict[(node_name, truncated)] = False
            node = self.nodes_dict[node_name]
            if truncated:
                reward_dict[(node_name, truncated)] = self.reward_kinds['col']
                termination_dict[(node_name, truncated)] = True
            else:
                if (node.x, node.y) == agent.goal_pos:
                    reward_dict[(node_name, truncated)] = self.reward_kinds['goal']
                    termination_dict[(node_name, truncated)] = True
                else:
                    reward_dict[(node_name, truncated)] = self.reward_kinds['step']
                    termination_dict[(node_name, truncated)] = False
        if self.step_count >= self.max_steps:
            for (node_name, truncated), prob in dynamics_dict.items():
                termination_dict[(node_name, truncated)] = True
        return reward_dict, termination_dict

    def take_actions(self, actions):
        for agent in self.env_agents:
            if not agent.terminated and not agent.truncated:
                a_action = actions[agent.name]
                a_pos = agent.pos
                pos_name = f'{a_pos[0]}_{a_pos[1]}'

                dynamics_dict = self.get_next_possible_nodes(pos_name, a_action)
                possible_poses, probs = [], []
                for (next_pos, truncated), prob in dynamics_dict.items():
                    possible_poses.append((next_pos, truncated))
                    probs.append(prob)
                reward_dict, termination_dict = self.get_next_possible_rewards(agent.name, dynamics_dict)
                # chosen_node_name, truncated = random.choices(list(dynamics_dict.keys()), weights=dynamics_dict.values(), k=1)[0]
                chosen_node_name, truncated = random.choices(possible_poses, weights=probs)[0]
                chosen_node = self.nodes_dict[chosen_node_name]

                agent.reward = reward_dict[(chosen_node_name, truncated)]
                agent.terminated = termination_dict[(chosen_node_name, truncated)]
                agent.truncated = truncated
                agent.pos = (chosen_node.x, chosen_node.y)

    def step(self, actions):
        """
        :return: observations, rewards, terminated, truncated, info
        """
        next_observations, rewards, terminated, truncated, info = {}, {}, {}, {}, {}
        self.take_actions(actions)
        self.step_count += 1
        for agent in self.env_agents:
            next_observations[agent.name] = self.build_obs(agent.pos)
            rewards[agent.name] = agent.reward
            terminated[agent.name] = agent.terminated
            truncated[agent.name] = agent.truncated

        return next_observations, rewards, terminated, truncated, info

    @staticmethod
    def close():
        plt.show()

    def render(self, info):
        plot_field(self.ax[0], info={
            'height': self.height,
            'width': self.width,
            'nodes': self.nodes,
            'env_agents': self.env_agents,
        })
        if 'value_graph' in info:
            plot_value_function(self.ax[1], info=info['value_graph'])
        plt.pause(0.001)


def main():
    img_dir = 'empty_4x4.map'  # 4-4
    nodes, nodes_dict, height, width = build_graph_nodes(img_dir=img_dir, path='../maps', show_map=False)
    square_env = SquareEnv(start_positions=[(1, 1)], goal_positions=[(4, 4)], nodes=nodes, nodes_dict=nodes_dict,
                           height=height, width=width)

    for i_run in range(1000):
        square_env.reset()
        termination_list = []
        step = 0
        while True not in termination_list:
            print(f'\r| run {i_run} | step {step} |', end='')
            actions = square_env.sample_actions()
            next_observations, rewards, terminated, truncated, info = square_env.step(actions)

            termination_list = []
            termination_list.extend(terminated.values())
            termination_list.extend(truncated.values())
            step += 1

            square_env.render(info={})

    square_env.close()


if __name__ == '__main__':
    main()
