import matplotlib.pyplot as plt
from networkx import nx
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def visualize(environment):
    G = nx.from_numpy_matrix(environment.state_space)
    print(G.edges(data=True))

    color_map = []
    for node in G.nodes():

        if node in environment.fire_locations:
            color_map.append('red')
        elif node in environment.terminal_states:
            color_map.append('green')
        else:
            color_map.append('cyan')


    nx.draw(G, with_labels=True, node_color=color_map, node_size=[100 for v in G.nodes()], layout=nx.spring_layout(G,scale=1000) )
    plt.savefig(os.path.join(dir_path, "environment_graph.pdf"))
    plt.savefig(os.path.join(dir_path, "environment_graph.png"))

