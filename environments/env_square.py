import random

from nodes_from_pic import build_graph_nodes
from globals import *


class EnvAgent:
    def __init__(self, start_pos, goal_pos, n_agent):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.n_agent = n_agent
        self.name = f'agent_{n_agent}'
        self.pos = self.start_pos
        self.actions = [0, 1, 2, 3, 4]  # 0 - up, 1 - right, 2 - down, 3 - left, 4 - stay
        self.terminated = False
        self.truncated = False
        self.reward = 0


def create_env_agents(start_positions, goal_positions):
    agents = []
    for n_agent, (start_pos, goal_pos) in enumerate(zip(start_positions, goal_positions)):
        agents.append(EnvAgent(start_pos, goal_pos, n_agent))
    return agents


class SquareEnv:
    def __init__(self, start_positions: list, goal_positions: list, nodes: list, nodes_dict: dict, height: int, width: int):
        self.start_positions = start_positions
        self.goal_positions = goal_positions
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.height = height
        self.width = width

        self.env_agents = None
        self.n_agents = None
        self.fig, self.ax = plt.subplots(2, 1)
        self.reward_kinds = {'col': -100, 'goal': 100, 'step': -1}

    def reset(self):
        self.env_agents = create_env_agents(self.start_positions, self.goal_positions)
        self.n_agents = len(self.env_agents)
        observations, rewards, terminated, truncated, info = {}, {}, {}, {}, {}
        for agent in self.env_agents:
            observations[agent.name] = agent.pos
            rewards[agent.name] = -1
            terminated[agent.name] = False
            truncated[agent.name] = False

        return observations, rewards, terminated, truncated, info

    def sample_actions(self):
        return {agent.name: random.choice(agent.actions) for agent in self.env_agents}

    def take_actions(self, actions):
        for agent in self.env_agents:
            if not agent.terminated and not agent.truncated:
                a_action = actions[agent.name]
                a_pos = agent.pos
                a_pos_node = self.nodes_dict[f'{a_pos[0]}_{a_pos[1]}']

                # 0 - up, 1 - right, 2 - down, 3 - left, 4 - stay
                straight = a_pos_node
                to_the_right = a_pos_node
                to_the_left = a_pos_node
                if a_action == 0:  # up
                    straight = a_pos_node.up_node
                    to_the_right = a_pos_node.right_node
                    to_the_left = a_pos_node.left_node
                elif a_action == 1:  # right
                    straight = a_pos_node.right_node
                    to_the_right = a_pos_node.down_node
                    to_the_left = a_pos_node.up_node
                elif a_action == 2:  # down
                    straight = a_pos_node.down_node
                    to_the_right = a_pos_node.left_node
                    to_the_left = a_pos_node.right_node
                elif a_action == 3:  # left
                    straight = a_pos_node.left_node
                    to_the_right = a_pos_node.up_node
                    to_the_left = a_pos_node.down_node
                chosen_node = random.choices([to_the_left, straight, to_the_right], weights=[0.1, 0.8, 0.1])[0]
                if chosen_node is None:
                    agent.reward = self.reward_kinds['col']
                    agent.truncated = True
                    return
                elif (chosen_node.x,  chosen_node.y) == agent.goal_pos:
                    agent.reward = self.reward_kinds['goal']
                    agent.terminated = True
                else:
                    agent.reward = self.reward_kinds['step']

                agent.pos = (chosen_node.x, chosen_node.y)

    def step(self, actions):
        """
        :return: observations, rewards, terminated, truncated, info
        """
        next_observations, rewards, terminated, truncated, info = {}, {}, {}, {}, {}
        self.take_actions(actions)
        for agent in self.env_agents:
            next_observations[agent.name] = agent.pos
            rewards[agent.name] = agent.reward
            terminated[agent.name] = agent.terminated
            truncated[agent.name] = agent.truncated

        return next_observations, rewards, terminated, truncated, info

    @staticmethod
    def close():
        plt.show()

    def render(self, info):
        self.plot_field(self.ax[0])
        plt.pause(0.001)

    def plot_field(self, ax):
        ax.cla()
        field = np.zeros((self.height, self.width))

        # map
        for node in self.nodes:
            field[node.x, node.y] = 1

        # start positions + goal positions + current positions
        for agent in self.env_agents:
            field[agent.start_pos[0], agent.start_pos[1]] = -1
            field[agent.goal_pos[0], agent.goal_pos[1]] = 2
            agent_circle = plt.Circle((agent.pos[0], agent.pos[1]), 0.2, color='r')
            ax.add_patch(agent_circle)

        # show
        ax.imshow(field, origin='lower')  # , cmap='gray'
        # circle1 = plt.Circle((0, 0), 0.2, color='r')
        # ax.add_patch(circle1)
        # ax.invert_yaxis()
        # ax.plot([i for i in range(self.side_size)])
        # ax.set_xlim(0, self.side_size)
        # ax.set_ylim(0, self.side_size)


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
