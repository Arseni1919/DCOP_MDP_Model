import copy

from globals import *


def plot_field(ax, info):
    ax.cla()
    field = np.zeros((info['height'], info['width']))

    # map
    for node in info['nodes']:
        field[node.x, node.y] = 1

    # start positions + goal positions + current positions
    colors = ['r', 'b', 'g', 'k', 'o']
    for agent in info['env_agents']:
        field[agent.start_pos[0], agent.start_pos[1]] = -1
        field[agent.goal_pos[0], agent.goal_pos[1]] = 2
        agent_circle = plt.Circle((agent.pos[1], agent.pos[0]), 0.2, color=colors[agent.n_agent % len(colors)])
        ax.add_patch(agent_circle)

    # show
    ax.imshow(field, origin='lower')  # , cmap='gray'
    # circle1 = plt.Circle((0, 0), 0.2, color='r')
    # ax.add_patch(circle1)
    # ax.invert_yaxis()
    # ax.plot([i for i in range(self.side_size)])
    # ax.set_xlim(0, self.side_size)
    # ax.set_ylim(0, self.side_size)


def plot_policy_function(ax, info):
    ax.cla()
    agent_name = info['name']
    field = np.zeros((info['height'], info['width']))
    node_dict = info['node_dict']
    # map
    for node_name, node_v in info['v_func'][agent_name].items():
        node = node_dict[node_name]
        field[node.x, node.y] = node_v

    # 1 - up, 2 - right, 3 - down, 4 - left, 0 - stay
    actions = {1: '^', 2: '>', 3: 'v', 4: '<', 0: '.'}
    for node_name, action in info['policy'][agent_name].items():
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


def plot_value_function(ax, info):
    ax.cla()
    agent_name = info['name']
    field = np.zeros((info['height'], info['width']))
    node_dict = info['node_dict']
    # map
    for node_name, node_v in info['v_func'][agent_name].items():
        node = node_dict[node_name]
        field[node.x, node.y] = node_v
        ax.text(node.y, node.x, f'{node_v: .2f}', ha="center", va="center", color="w", size=3)

    # show
    ax.imshow(field, origin='lower')  # , cmap='gray' , origin='lower'


def plot_value_function_united(ax, info):
    ax.cla()
    v_func_1 = info['v_func']['agent_0']
    v_func_2 = copy.deepcopy(info['v_func']['agent_1'])
    for state_name, state_value in v_func_1.items():
        v_func_2[state_name] -= state_value
    field = np.zeros((info['height'], info['width']))
    node_dict = info['node_dict']
    # map
    for node_name, node_v in v_func_2.items():
        node = node_dict[node_name]
        field[node.x, node.y] = node_v
        ax.text(node.y, node.x, f'{node_v: .2f}', ha="center", va="center", color="w", size=3)

    # show
    ax.imshow(field, origin='lower')  # , cmap='gray' , origin='lower'


def plot_var_func_nodes(ax, info):
    ax.cla()
    var_nodes_list = info['var_nodes_list']
    func_nodes_list = info['func_nodes_list']

    # plot links between nodes
    for pair in combinations(var_nodes_list, 2):
        var1, var2 = pair[0], pair[1]
        ax.plot([var1.x, var2.x], [var1.y, var2.y])

    # plot actions
    for var in var_nodes_list:
        new_x, new_y, dx, dy = var.action_dir()
        ax.arrow(var.x, var.y, dx, dy, width=0.5, head_width=1)

    ax.set_title('Var and Func Nodes')
    radius = var_nodes_list[0].radius
    ax.set_xlim(0 - radius, 100 + radius)
    ax.set_ylim(0 - radius, 100 + radius)


def plot_func_node(ax, info):
    ax.cla()
    func_node = info['func_node']

    field = func_node.func
    if field.shape == (5, 5):
        names = ['still', 'up', 'right', 'down', 'left']
        ax.set_xticks(np.arange(5), labels=names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(5), labels=names)
    # show
    # for node_name, node_v in info['v_func'][agent_name].items():
    #     node = node_dict[node_name]
    #     field[node.x, node.y] = node_v
    #     ax.text(node.y, node.x, f'{node_v: .2f}', ha="center", va="center", color="w", size=3)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            text = ax.text(j, i, f'{field[i, j]: .2f}', ha="center", va="center", color="k", size=5)
    ax.imshow(field, origin='lower')  # , cmap='gray' , origin='lower'


def plot_total_cost(ax, info):
    ax.cla()
    iterations = info['iterations']
    total_costs = info['total_costs']
    ax.plot(total_costs)
    ax.set_xlim(0, iterations)
    ax.set_title('Total Costs')


def plot_actions(ax, info):
    ax.cla()
    var_nodes_list = info['var_nodes_list']
    func_nodes_list = info['func_nodes_list']
    iterations = info['iterations']
    counter = 0
    for var in var_nodes_list:
        ax.plot(var.action_history, label=var.name)
        counter += 1
        if counter > 2:
            break
    if var_nodes_list[0].n_actions == 5:
        names = ['still', 'up', 'right', 'down', 'left']
        ax.set_yticks(np.arange(5), labels=names)
    ax.legend()
    ax.set_xlim(0, iterations)
    ax.set_title('Actions')

