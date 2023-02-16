from globals import *
# from funcs_graph.map_dimensions import map_dimensions_dict
from simulator_objects import Node
import re
import math


def get_dims(img_dir, path='maps'):
    if '.map' in img_dir:
        with open(f'{path}/{img_dir}') as f:
            lines = f.readlines()
            height = int(re.search(r'\d+', lines[1]).group())
            width = int(re.search(r'\d+', lines[2]).group())
        return height, width
    else:
        raise RuntimeError('not a .map format')


def get_np_from_dot_map(img_dir, path='maps'):
    with open(f'{path}/{img_dir}') as f:
        lines = f.readlines()
        height, width = get_dims(img_dir, path)
        img_np = np.zeros((height, width))
        for height_index, line in enumerate(lines[4:]):
            for width_index, curr_str in enumerate(line):
                if curr_str == '.':
                    img_np[height_index, width_index] = 1
        return img_np, (height, width)


def build_graph_nodes(img_dir, path='maps', show_map=False):
    print('Starts to build_graph_nodes...')
    if '.map' in img_dir:
        img_np, (height, width) = get_np_from_dot_map(img_dir, path)
    else:
        raise RuntimeError('format of the map is not supported')
    return build_graph_from_np(img_np, show_map, height, width)


def distance_nodes(node1, node2, h_func: dict = None):
    if h_func is None:
        # print('regular distance')
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
    else:
        heuristic_dist = h_func[node1.x][node1.y][node2.x][node2.y]
        # direct_dist = np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
        return heuristic_dist


def set_nei(name_1, name_2, nodes_dict):
    if name_1 in nodes_dict and name_2 in nodes_dict and name_1 != name_2:
        node1 = nodes_dict[name_1]
        node2 = nodes_dict[name_2]
        dist = distance_nodes(node1, node2)
        if dist == 1:
            node1.neighbours.append(node2.xy_name)
            node2.neighbours.append(node1.xy_name)


def update_orientation_nodes_parameters(nodes, nodes_dict):
    for node in nodes:

        # node.up_node = node
        # node.right_node = node
        # node.down_node = node
        # node.left_node = node

        x, y = node.x, node.y
        location_name = f'{x}_{y + 1}'
        if location_name in nodes_dict:
            node.up_node = nodes_dict[location_name]
        location_name = f'{x + 1}_{y}'
        if location_name in nodes_dict:
            node.right_node = nodes_dict[location_name]
        location_name = f'{x}_{y - 1}'
        if location_name in nodes_dict:
            node.down_node = nodes_dict[location_name]
        location_name = f'{x - 1}_{y}'
        if location_name in nodes_dict:
            node.left_node = nodes_dict[location_name]


def make_self_neighbour(nodes):
    for node_1 in nodes:
        node_1.neighbours.append(node_1.xy_name)


def make_neighbours(nodes):
    for node_1 in nodes:
        node_1.neighbours.append(node_1.xy_name)
        for node_2 in nodes:
            if node_1.xy_name != node_2.xy_name:
                dist = math.sqrt((node_1.x - node_2.x)**2 + (node_1.y - node_2.y)**2)
                if dist == 1.0:
                    node_1.neighbours.append(node_2.xy_name)


def build_graph_from_np(img_np, show_map=False, height=None, width=None):
    # 0 - wall, 1 - free space
    nodes = []
    nodes_dict = {}

    x_size, y_size = img_np.shape
    # CREATE NODES
    for i_x in range(x_size):
        for i_y in range(y_size):
            if img_np[i_x, i_y] == 1:
                node = Node(i_x, i_y)
                nodes.append(node)
                nodes_dict[node.xy_name] = node

    # CREATE NEIGHBOURS
    # make_neighbours(nodes)

    name_1, name_2 = '', ''
    for i_x in range(x_size):
        for i_y in range(y_size):
            name_2 = f'{i_x}_{i_y}'
            set_nei(name_1, name_2, nodes_dict)
            name_1 = name_2

    print('finished rows')

    for i_y in range(y_size):
        for i_x in range(x_size):
            name_2 = f'{i_x}_{i_y}'
            set_nei(name_1, name_2, nodes_dict)
            name_1 = name_2
    make_self_neighbour(nodes)
    print('finished columns')

    # orientation variables
    update_orientation_nodes_parameters(nodes, nodes_dict)

    if show_map:
        plt.imshow(img_np, cmap='gray', origin='lower')
        # plt.gca().invert_yaxis()
        plt.show()
        # plt.pause(1)
        # plt.close()

    return nodes, nodes_dict, height, width


def main():
    img_dir = 'empty_4x4.map'  # 64-64
    # img_dir = 'room-64-64-8.map'  # 64-64
    # img_dir = 'warehouse-10-20-10-2-1.map'  # 63-161
    # img_dir = 'warehouse-10-20-10-2-2.map'  # 84-170
    # img_dir = 'warehouse-20-40-10-2-1.map'  # 123-321
    # img_dir = 'ht_chantry.map'  # 141-162
    # img_dir = 'lt_gallowstemplar_n.map'  # 180-251
    # img_dir = 'lak303d.map'  # 194-194
    # img_dir = 'warehouse-20-40-10-2-2.map'  # 164-340
    # img_dir = 'Berlin_1_256.map'  # 256-256
    # img_dir = 'den520d.map'  # 257-256
    # img_dir = 'ht_mansion_n.map'  # 270-133
    # img_dir = 'brc202d.map'  # 481-530
    nodes, nodes_dict, height, width = build_graph_nodes(img_dir=img_dir, path='../maps', show_map=True)
    print()


if __name__ == '__main__':
    main()


