import random
import copy
import numpy as np

from globals import *
from environments.env_square import SquareEnv
from environments.nodes_from_pic import build_graph_nodes


def get_policy_from_values(env, agent_name, values_of_states):
    agent = env.get_agent(agent_name)
    policy = {}
    for state_name, value in values_of_states.items():
        action_values_dict = {}
        for action in agent.actions:
            straight, _, _ = env.get_nodes_after_action(state_name, action)
            if straight is not None:
                action_values_dict[action] = values_of_states[straight.xy_name]
            else:
                pos = agent.pos
                action_values_dict[action] = values_of_states[f'{pos[0]}_{pos[1]}']
        policy[state_name] = max(action_values_dict, key=action_values_dict.get)
    return policy


def value_iteration(env, agent_name='agent_0'):
    const_delta_limit = 0.001
    const_lambda = 0.9

    delta = 100
    values_of_states = env.get_states()
    policy = {state: env.sample_action(agent_name) for state in values_of_states.keys()}

    iters = 0
    while delta > const_delta_limit:
        # print(f'\rvalue_iteration step - {iters}', end='')
        print(f'value_iteration step - {iters} | delta: {delta}')
        iters += 1
        delta = 0
        for state, _ in policy.items():
            v_state = values_of_states[state]
            # choose max action
            agent = env.get_agent(agent_name)

            action_values_dict = {}
            for action in agent.actions:
                dynamics_dict = env.get_next_possible_nodes(state, action)
                reward_dict, termination_dict = env.get_next_possible_rewards(agent_name, dynamics_dict)
                curr_action_value = 0
                for (next_state, truncated), prob in dynamics_dict.items():
                    reward = reward_dict[(next_state, truncated)]
                    curr_action_value += prob * (reward + const_lambda * values_of_states[next_state])

                    # termination = termination_dict[(next_state, truncated)]
                    # if termination:
                    #     curr_action_value += prob * reward
                    # else:
                    #     curr_action_value += prob * (reward + const_lambda * values_of_states[next_state])

                action_values_dict[action] = curr_action_value
            values_of_states[state] = max(action_values_dict.values())
            # policy[state] = max(action_values_dict, key=action_values_dict.get)
            delta = max(delta, abs(v_state - values_of_states[state]))

    print()
    return policy, values_of_states


def main():
    # img_dir = 'empty_4x4.map'  # 4-4
    img_dir = 'empty_XxX.map'  # x-x
    nodes, nodes_dict, height, width = build_graph_nodes(img_dir=img_dir, path='../maps', show_map=False)
    # s_g_nodes = random.sample(nodes, 2)
    square_env = SquareEnv(
        start_positions=[(3, 3), (3, 7)],
        goal_positions=[(7, 7), (7, 3)],
        nodes=nodes, nodes_dict=nodes_dict,
        height=height, width=width
    )
    square_env.reset()
    agents = square_env.env_agents
    agents_policies = {}
    agents_values = {}
    for agent in agents:
        policy, values_of_states = value_iteration(env=square_env, agent_name=agent.name)
        agents_policies[agent.name] = policy
        agents_values[agent.name] = values_of_states

    # slight change
    v_func_0 = agents_values['agent_0']
    v_func_1 = agents_values['agent_1']

    # v_func_2 = copy.deepcopy(agents_values['agent_1'])

    for state_name, state_value in v_func_0.items():
        v_func_1[state_name] -= 0.2 * state_value

    agents_policies['agent_0'] = get_policy_from_values(square_env, 'agent_0', v_func_0)
    agents_policies['agent_1'] = get_policy_from_values(square_env, 'agent_1', v_func_1)

    for i_run in range(1000):
        obs, rewards, terminated, truncated, info = square_env.reset()
        termination_list = []
        step = 0
        while True not in termination_list:
            print(f'\r| run {i_run} | step {step} |', end='')
            actions = {}
            for agent in agents:
                agent_obs = obs[agent.name]
                actions[agent.name] = agents_policies[agent.name][agent_obs]
            next_obs, rewards, terminated, truncated, info = square_env.step(actions)

            termination_list = []
            termination_list.append(all(terminated.values()))
            termination_list.append(any(truncated.values()))
            step += 1
            obs = next_obs

            square_env.render(info={
                'policy_graph_0': {
                    'name': 'agent_0',
                    'height': square_env.height,
                    'width': square_env.width,
                    'node_dict': square_env.nodes_dict,
                    'v_func': agents_values,
                    'policy': agents_policies,
                },
                'policy_graph_1': {
                    'name': 'agent_1',
                    'height': square_env.height,
                    'width': square_env.width,
                    'node_dict': square_env.nodes_dict,
                    'v_func': agents_values,
                    'policy': agents_policies
                },
                'value_graph_0': {
                    'name': 'agent_0',
                    'height': square_env.height,
                    'width': square_env.width,
                    'node_dict': square_env.nodes_dict,
                    'v_func': agents_values,
                    'policy': agents_policies,
                },
                'value_graph_1': {
                    'name': 'agent_1',
                    'height': square_env.height,
                    'width': square_env.width,
                    'node_dict': square_env.nodes_dict,
                    'v_func': agents_values,
                    'policy': agents_policies
                },
                # 'value_graph_united': {
                #     'height': square_env.height,
                #     'width': square_env.width,
                #     'node_dict': square_env.nodes_dict,
                #     'v_func': agents_values,
                #     'policy': agents_policies
                # }
            })

    square_env.close()


if __name__ == '__main__':
    main()



