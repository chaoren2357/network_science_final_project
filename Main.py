# -*- coding: utf-8 -*-
"""
@author: Zhiyuan Gao
"""

# enviorment dependency
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import pylab
import time
import pickle

# plt.rcParams['savefig.dpi'] = 1000 #图片像素
# plt.rcParams['figure.dpi'] = 1000 #分辨率
# plt.rcParams['axes.unicode_minus'] = False

import os
def heatmap(data, row_labels, col_labels, ax=None,cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
def read_file(filename,source = " "):
    """Read file and make a networkx graph class

    Parameters
    ---------------------
    filename: str
        The name of data file, direct path
    source: str (default: "Network Repository")
        The resource of data, now only avaliable for ["SNAP", "Network Repository"]
    Returns:
    ----------------------
    G: nx.Graph()
        The graph build by data in file
    """
    G = nx.Graph()
    if source == "SNAP": # data from Stanford Large Network Dataset Collection
        with open(filename) as f:
            for line in f:
                if line[0] == "#":
                    pass
                else:
                    from_str,to_str = line.split('\t',1)
                    from_num = int(from_str)
                    to_num = int(to_str)
                    try:
                        G.add_edge(from_num,to_num)
                    except:
                        print("This line has some problem: ",line)
    elif source == "Network Repository":
        with open(filename) as f:
            for line in f:
                if line[0] == "%" :
                    pass
                else:
                    from_str,to_str = line.split(' ',1)
                    from_num = int(from_str)
                    to_num = int(to_str)
                    try:
                        G.add_edge(from_num,to_num)
                    except:
                        print("This line has some problem: ",line)
    else:
        raise ValueError("Please check the source of your data")
    return G
def get_basic_information(G,name,savepath):
    """Get basic information of a Graph

     Parameters
     ---------------------
     G: nx.Graph()
         The graph to be described
     show: bool (default: False)
         Whether print the result or not

     Returns:
     ----------------------
     dict: dictionary
         A dictionary contains information includes:
         { NN: Number of nodes;
           NE: Number of edges;
           MD: Maximum degree;
           AD: Average degree;
           DIA: Diameter(longest shortest path);
           AC: Average clustering coefficient;
           NT: Numer of triangles;
         }
     """
    dict = {}
    s = time.time()

    dict["NN"] = ("Number of nodes",G.number_of_nodes())
    e1 = time.time()
    print("NN:{0},total:{1}".format(format(e1-s,".2f"),format(e1-s,".2f")))

    dict["NE"] = ("Number of edges",G.number_of_edges())
    e2 = time.time()
    print("NE:{0},total:{1}".format(format(e2-e1,".2f"),format(e2-s,".2f")))

    dict["DH"] = ("Degree Histogram",nx.degree_histogram(G))
    e3 = time.time()
    print("DH:{0},total:{1}".format(format(e3-e2,".2f"),format(e3-s,".2f")))

    try:
        dict["DIA"] = ("Diameter",nx.diameter(G))
    except:
        dict["DIA"] = ("Diameter","INF")
    e4 = time.time()
    print("DIA:{0},total:{1}".format(format(e4-e3,".2f"),format(e4-s,".2f")))

    dict["C"] = ("Clustering coefficient",nx.clustering(G))
    e5 = time.time()
    print("C:{0},total:{1}".format(format(e5-e4,".2f"),format(e5-s,".2f")))

    dict["T"] = ("Triangles",nx.triangles(G))
    e6 = time.time()
    print("T:{0},total:{1}".format(format(e6-e5,".2f"),format(e6-s,".2f")))

    dict["D"] = ("Digree",nx.degree(G))
    e7 = time.time()
    print("D:{0},total:{1}".format(format(e7-e6,".2f"),format(e7-s,".2f")))

    dict["E"] = ("Edges",nx.edges(G))
    e8 = time.time()
    print("NN:{0},total:{1}".format(format(e8-e7,".2f"),format(e8-s,".2f")))

    # dict["B"] = ("Betweenness",nx.betweenness_centrality(G))
    # e9 = time.time()
    # print("B:{0},total:{1}".format(e9-e8,e9-s))

    with open(os.path.join(savepath, name + ".pkl"),
              'wb') as fo:  # 将数据写入pkl文件
        pickle.dump(dict, fo)
def add_sth(G,name,savepath):
    dict = {}
    s = time.time()
def show_basic_information(dict):
    # Number of nodes
    # Number of edges
    # Diameter
    for key in dict:
       if key in ["NN","NE","DIA"]:
           print("{0}:{1}".format(dict[key][0],dict[key][1]))

    # Maximum degree
    # Average degree
    print("Maximum degree:{0}".format(len(dict["DH"][1])-1))
    print("Average degree:{0}".format(2*dict["NE"][1]/dict["NN"][1]))

    # Average clustering coefficient
    # Numer of triangles
    print("Average clustering coefficient:{0}".format(sum([dict["C"][1][k] for k in dict["C"][1]])/dict["NN"][1]))
    print("Numer of triangles:{0}".format(sum([dict["T"][1][node] for node in dict["T"][1]])/3))


def plot_degree_distribution(dicts,savepath,names,flag="bar"):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("k")
    ax1.set_ylabel(r"$P_k$",)
    width = 0.9 / len(names)

    # ax2 = ax1.twinx()
    # ax2.set_ylabel(r"$P_k$", color='tab:red')  # we already handled the x-label with ax1
    # ax2.set_yscale("log")
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    for idx,(name,dict) in enumerate(zip(names,dicts)):
        p = np.array(dict["DH"][1])/dict["NN"][1]
        if flag == "bar":
            ax1.bar([i + idx * width for i in range(len(p))], p, width=width, label=name)
        elif flag == "log":
            ax1.set_yscale("log")
            ax1.bar([i + idx * width for i in range(len(p))], p, width=width, label=name)

        # ax2.plot([i for i in range(len(p))],p,color = 'red',label=name)

    fig.tight_layout()
    plt.title("Digree Distribution")
    plt.legend()
    plt.savefig(os.path.join(savepath,"_".join(names)+"_digree_distribution_"+flag+".png"))
    print("Finish plotting {0}".format("_".join(names)+"_digree_distribution_"+flag+".png"))

def save_graph(graph,file_name):
    #initialze Figure
    plt.figure(num=None, figsize=(1000, 1000), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.kamada_kawai_layout(graph)
    nx.draw_networkx_nodes(graph,pos,node_size=100000)
    nx.draw_networkx_edges(graph,pos,width=10)
    nx.draw_networkx_labels(graph,pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    xmin = cut * min(xx for xx, yy in pos.values())
    ymin = cut * min(yy for xx, yy in pos.values())
    print(xmin,xmax,ymin,ymax)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.savefig(file_name,bbox_inches="tight")
    pylab.close()
    del fig

def plot_degree_correlations(dicts,savepath,names):
    for idx, (name, dict) in enumerate(zip(names, dicts)):
        k_num = len(dict["DH"][1])-1
        labels = np.arange(k_num)+1
        e_num=len(dict["E"][1])
        print("E_num:{0}".format(e_num))
        fig, ax = plt.subplots()

        prob = np.zeros((k_num, k_num))
        for i, j in dict["E"][1]:
            prob[dict["D"][1][i] - 1][dict["D"][1][j] - 1] = prob[dict["D"][1][i] - 1][dict["D"][1][j] - 1] + 1
        prob = prob / len(dict["E"][1])
        im, cbar = heatmap(prob, row_labels=labels, col_labels=labels, ax=ax,
                           cmap="YlGn", cbarlabel="degree_corr")
        fig.tight_layout()
        plt.show()

# filenames = ["road-euroroad.edges","road-luxembourg-osm.mtx","road-belgium-osm.mtx","road-germany-osm.mtx"]#
# filenames = ["road-chesapeake.mtx"]#
filenames = ["road-euroroad.edges"]#

dicts = []
for filename in filenames:
    print("----------------------------------------------------------------------")
    print("For {0}".format(filename))
    filepath = os.path.join("/Users/scott/Desktop/Classes/NetworkScience/Data",filename)
    name = filename.split(".",1)[0]
    save_path = "/Users/scott/Desktop"
    G=read_file(filepath,"Network Repository") #["SNAP", "Network Repository"]
    # get_basic_information(G,name,save_path)
    save_graph(G,"/Users/scott/Desktop/save.pdf")
    # print("------------------Results-------------------")
    path = os.path.join(save_path, name + ".pkl")
    dict = pickle.load(open(path, 'rb'))
    # show_basic_information(name,save_path)
    dicts.append(dict)
# plot_degree_distribution(dicts,save_path,filenames,"bar")
# plot_degree_distribution(dicts,save_path,filenames,"log")
# plot_degree_correlations(dicts,save_path,filenames)