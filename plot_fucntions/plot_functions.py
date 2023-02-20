from globals import *


def plot_field(ax, info):
    ax.cla()
    field = np.zeros((info['height'], info['width']))

    # map
    for node in info['nodes']:
        field[node.x, node.y] = 1

    # start positions + goal positions + current positions
    for agent in info['env_agents']:
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


def plot_value_function(ax, info):
    ax.cla()
    field = np.zeros((info['height'], info['width']))
    node_dict = info['node_dict']
    # map
    for node_name, node_v in info['v_func'].items():
        node = node_dict[node_name]
        field[node.x, node.y] = node_v

    # 1 - up, 2 - right, 3 - down, 4 - left, 0 - stay
    actions = {1: '^', 2: '>', 3: 'v', 4: '<', 0: '.'}
    for node_name, action in info['policy'].items():
        node = node_dict[node_name]
        ax.text(node.x, node.y, actions[action],
                ha="center", va="center", color="w")

    # show
    ax.imshow(field, origin='lower')  # , cmap='gray' , origin='lower'
    # circle1 = plt.Circle((0, 0), 0.2, color='r')
    # ax.add_patch(circle1)
    # ax.invert_yaxis()
    # ax.plot([i for i in range(self.side_size)])
    # ax.set_xlim(0, self.side_size)
    # ax.set_ylim(0, self.side_size)

