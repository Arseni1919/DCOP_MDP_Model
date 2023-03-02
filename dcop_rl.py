import math
import random

import matplotlib.pyplot as plt
import numpy as np

from plot_fucntions.plot_functions import *
from globals import *


class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(3, 2)  # , gridspec_kw={'height_ratios': [3, 1, 2, 1]}

    def render(self, info):
        plot_var_func_nodes(self.ax[0, 0], info)
        # var_nodes_list = info['var_nodes_list']
        func_nodes_list = info['func_nodes_list']
        counter = 0
        for func in func_nodes_list:
            plot_func_node(self.ax[counter, 1], info={'func_node': func})
            counter += 1
            if counter > 2:
                break

        if 'iterations' in info:
            plot_total_cost(self.ax[1, 0], info)
            plot_actions(self.ax[2, 0], info)
        plt.pause(0.001)


class VarNodeRL:
    def __init__(self, num, x=0, y=0):
        self.num = num
        self.x = x
        self.y = y
        self.name = f'var_{self.num}'
        self.var_nei_l = []
        self.var_nei_d = {}
        self.func_nei_l = []
        self.func_nei_d = {}
        self.radius = 4
        self.n_actions = 5
        self.domain = list(range(self.n_actions))
        self.action = random.choice(self.domain)
        self.actions_weights = [1/self.n_actions for _ in self.domain]
        self.action_history = [self.action]

        self.curr_reward = 0

    def add_var_nei(self, var_nei):
        self.var_nei_l.append(var_nei)
        self.var_nei_d[var_nei.name] = var_nei

    def add_func_nei(self, func_nei):
        self.func_nei_l.append(func_nei)
        self.func_nei_d[func_nei.name] = func_nei

    def action_dir(self):
        # teta = 2 * self.n_actions * np.pi / (self.action + 1)
        # new_x = self.radius * np.cos(teta) + self.x
        # dx = new_x - self.x
        # new_y = self.radius * np.sin(teta) + self.y
        # dy = new_y - self.y

        new_x, new_y, dx, dy = self.x, self.y, 0, 0
        if self.action == 1:  # up
            new_x, new_y, dx, dy = self.x, self.y + self.radius, 0, self.radius
        if self.action == 2:  # right
            new_x, new_y, dx, dy = self.x + self.radius, self.y, self.radius, 0
        if self.action == 3:  # down
            new_x, new_y, dx, dy = self.x, self.y - self.radius, 0, -self.radius
        if self.action == 4:  # left
            new_x, new_y, dx, dy = self.x - self.radius, self.y, -self.radius, 0

        return new_x, new_y, dx, dy

    def decide_action(self, dsa_prob):
        # if random.random
        # print(self.actions_weights)
        # action = np.random.choice(self.domain, 1, p=self.actions_weights)[0]
        # dsa_prob = min(dsa_prob, 0.95)
        if random.random() < dsa_prob:
            max_weight = np.max(self.actions_weights)
            action_indx = np.where(self.actions_weights == max_weight)[0]
        else:
            action_indx = self.domain
        self.action = np.random.choice(action_indx)
        self.action_history.append(self.action)

    def update_weights(self):
        costs = np.ones(self.n_actions) / 1000
        for func in self.func_nei_l:
            new_costs = func.build_new_costs(var=self)
            costs += new_costs
        new_probs = costs / np.sum(costs)
        self.actions_weights = new_probs
        # print([ '%.2f' % elem for elem in new_probs])


class FuncNodeRL:
    def __init__(self, num, var_nei_l):
        self.num = num
        self.var_nei_l = var_nei_l
        self.name = f'func'
        self.var_nei_d = {}
        for var in self.var_nei_l:
            self.name += f'_{var.num}'
            self.var_nei_d[var.name] = var
        self.func = np.random.rand(*[var.n_actions for var in self.var_nei_l])
        self.func_counter = np.ones(self.func.shape)

    def update_func(self, alpha=0.9):
        """
        curAvg = curAvg + (newNum - curAvg)/n
        """
        # print(self.func)
        action1 = self.var_nei_l[0].action
        action2 = self.var_nei_l[1].action
        value = self.func[action1, action2]
        value_count = self.func_counter[action1, action2]
        cost = self.var_nei_l[0].curr_reward + self.var_nei_l[1].curr_reward
        # print(self.func[action1, action2])
        # self.func[action1, action2] = value + (cost - value) / value_count
        self.func[action1, action2] = value + (cost - value) * alpha
        # print(self.func[action1, action2])
        self.func_counter[action1, action2] += 1

    def build_new_costs(self, var):
        index_of = self.var_nei_l.index(var)
        new_costs = np.zeros(var.n_actions)
        for action in range(var.n_actions):
            if index_of == 0:
                values_to_examine = self.func[action, :]
            elif index_of == 1:
                values_to_examine = self.func[:, action]
            else:
                raise RuntimeError()
            # min_cost = np.min(values_to_examine)
            # new_costs[action] += min_cost
            max_cost = np.max(values_to_examine)
            new_costs[action] += max_cost

        return new_costs


def clique_vars(var_nodes_list):
    for pair in combinations(var_nodes_list, 2):
        var1, var2 = pair[0], pair[1]
        var1.add_var_nei(var2)
        var2.add_var_nei(var1)


def create_func_nodes(var_nodes_list, func_nodes_list):
    counter = 0
    for pair in combinations(var_nodes_list, 2):
        var1, var2 = pair[0], pair[1]
        func_node = FuncNodeRL(counter, var_nei_l=[var1, var2])
        func_nodes_list.append(func_node)
        var1.add_func_nei(func_node)
        var2.add_func_nei(func_node)
        counter += 1


def reward_func(var):
    cost = 0
    new_x, new_y, _, _ = var.action_dir()
    for var_nei in var.var_nei_l:
        new_nei_x, new_nei_y, _, _ = var_nei.action_dir()
        distance = math.dist((new_x, new_y), (new_nei_x, new_nei_y))
        # cost += (1 / distance) * 100  # + random.random()
        # cost += np.log(distance)
        cost += distance
    # print(f'\n---\n{cost}\n---\n')
    return cost


def calc_total_cost(var_nodes_list):
    total = 0
    for var in var_nodes_list:
        total += var.curr_reward
    return total


def train(var_nodes_list, func_nodes_list, plotter, plot_every):
    """
    curAvg = curAvg + (newNum - curAvg)/n
    """
    iterations = 1000
    alpha = 0.9
    total_costs = []
    for iteration in range(iterations):
        dsa_prob = iteration/(0.8 * iterations)
        # decide on action
        for var in var_nodes_list:
            var.decide_action(dsa_prob=dsa_prob)

        # exchange actions with var neighbors - done

        # get reward
        for var in var_nodes_list:
            var.curr_reward = reward_func(var)

        # send rewards + actions to func neighbors - done

        # update func neighbors
        for func in func_nodes_list:
            func.update_func(alpha=alpha)

        # send weights from func to var
        for var in var_nodes_list:
            var.update_weights()

        # stats
        print(f'\r{dsa_prob=}', end='')
        total_costs.append(calc_total_cost(var_nodes_list))
        if iteration % plot_every == 0:
            plotter.render(info={
                'var_nodes_list': var_nodes_list,
                'func_nodes_list': func_nodes_list,
                'iterations': iterations,
                'total_costs': total_costs,
            })


def main():
    plotter = Plotter()
    plot_every = 100
    n_var_nodes = 50
    x_l = random.sample(range(100), n_var_nodes)
    y_l = random.sample(range(100), n_var_nodes)
    var_nodes_list = [VarNodeRL(num, x_l[num], y_l[num]) for num in range(n_var_nodes)]
    func_nodes_list = []
    clique_vars(var_nodes_list)
    create_func_nodes(var_nodes_list, func_nodes_list)

    train(var_nodes_list, func_nodes_list, plotter, plot_every)

    plotter.render(info={
        'var_nodes_list': var_nodes_list,
        'func_nodes_list': func_nodes_list
    })

    plt.show()


if __name__ == '__main__':
    main()
