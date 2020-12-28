# -*- coding: utf-8 -*-
"""
@author: Zhiyuan Gao
"""
"""Main program of analysing the network"""

from Utils import *
import os


## Names of dataset, mine is listed below
# ["road-chesapeake.mtx","road-euroroad.edges","road-usroads.mtx","road-roadNet-CA.mtx","road-germany-osm.mtx"]
filenames = ["road-chesapeake.mtx","road-euroroad.edges","road-usroads.mtx","road-roadNet-CA.mtx","road-germany-osm.mtx"]

dicts = []
for filename in filenames:
    print("----------------------------------------------------------------------")
    print("For {0}".format(filename))
    filepath = os.path.join("/Users/scott/Data/networkScience/origin",filename) # Path to your dataset
    name = filename.split(".",1)[0]
    save_path = "/Users/scott/Data/networkScience/pickle" # Path to your pickle file and other stuff
    # G=read_file(filepath,"Network Repository") #["SNAP", "Network Repository"]
    # get_basic_information(G,name,save_path)
    # add_betweenness(G,name,save_path)
    # save_graph(G,save_path,name)
    # print("------------------Results-------------------")
    path = os.path.join(save_path, name + ".pkl")
    dict = pickle.load(open(path, 'rb'))
    # show_basic_information(dict)
    dicts.append(dict)

# plot_degree_distribution(dicts,save_path,filenames,"bar")
# plot_degree_distribution(dicts,save_path,filenames,"log")
# plot_betweenness_distribution(dicts,save_path,filenames)
# plot_degree_betweenness_fig(dicts,save_path,filenames)
# plot_clustering_distribution(dicts,save_path,filenames,flag="log")
# plot_degree_clustering_fig(dicts,save_path,filenames)
# plot_degree_correlations(dicts,save_path,filenames)
# plot_triangle_distribution(dicts,save_path,filenames,"bar")
# plot_triangle_distribution(dicts,save_path,filenames,"log")
plot_degree_triangle_fig(dicts,save_path,filenames)


# # Diameter
# ll =  nx.diameter(G)
# # Shortest Path
# ave_l = nx.average_shortest_path_length(G)
#
# # Communities
# communities = sorted(nx.algorithms.community.greedy_modularity_communities(G),key=len,reverse=True)
