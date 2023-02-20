import random

import numpy as np

from globals import *
from environments.env_square import SquareEnv
from environments.nodes_from_pic import build_graph_nodes


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
                possible_poses = list(dynamics_dict.keys())
                reward_dict, termination_dict = env.get_next_possible_rewards(agent_name, possible_poses)
                curr_action_value = 0
                for next_state, prob in dynamics_dict.items():
                    reward = reward_dict[next_state]
                    termination = termination_dict[next_state]
                    if termination:
                        curr_action_value += prob * reward
                    else:
                        curr_action_value += prob * (reward + const_lambda * values_of_states[next_state])
                action_values_dict[action] = curr_action_value
            values_of_states[state] = max(action_values_dict.values())
            policy[state] = max(action_values_dict, key=action_values_dict.get)
            delta = max(delta, abs(v_state - values_of_states[state]))

    print()
    return policy, values_of_states


def main():
    # img_dir = 'empty_4x4.map'  # 4-4
    img_dir = 'empty_XxX.map'  # x-x
    nodes, nodes_dict, height, width = build_graph_nodes(img_dir=img_dir, path='../maps', show_map=False)
    # s_g_nodes = random.sample(nodes, 2)
    square_env = SquareEnv(
        start_positions=[(1, 1)],
        goal_positions=[(7, 7)],
        nodes=nodes, nodes_dict=nodes_dict,
        height=height, width=width
    )
    square_env.reset()
    agent_name = 'agent_0'
    policy, values_of_states = value_iteration(env=square_env, agent_name=agent_name)

    for i_run in range(1000):
        obs, rewards, terminated, truncated, info = square_env.reset()
        termination_list = []
        step = 0
        while True not in termination_list:
            print(f'\r| run {i_run} | step {step} |', end='')
            actions = {agent_name: policy[obs[agent_name]]}
            next_obs, rewards, terminated, truncated, info = square_env.step(actions)

            termination_list = []
            termination_list.extend(terminated.values())
            termination_list.extend(truncated.values())
            step += 1
            obs = next_obs

            square_env.render(info={'value_graph': {
                'height': square_env.height,
                'width': square_env.width,
                'node_dict': square_env.nodes_dict,
                'v_func': values_of_states,
                'policy': policy
            }})

    square_env.close()


if __name__ == '__main__':
    main()



