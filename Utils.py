import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import pylab
import time
import pickle
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
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

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

    dict["E"] = ("Edges",nx.edges(G))
    e4 = time.time()

    print("E:{0},total:{1}".format(format(e4-e3,".2f"),format(e4-s,".2f")))

    dict["C"] = ("Clustering coefficient",nx.clustering(G))
    e5 = time.time()
    print("C:{0},total:{1}".format(format(e5-e4,".2f"),format(e5-s,".2f")))

    dict["T"] = ("Triangles",nx.triangles(G))
    e6 = time.time()
    print("T:{0},total:{1}".format(format(e6-e5,".2f"),format(e6-s,".2f")))

    dict["D"] = ("Digree",nx.degree(G))
    e7 = time.time()
    print("D:{0},total:{1}".format(format(e7-e6,".2f"),format(e7-s,".2f")))

    dict["AD"] = ("Average Degree",nx.average_degree_connectivity(G))
    e8 = time.time()
    print("AD:{0},total:{1}".format(format(e8-e7,".2f"),format(e8-s,".2f")))

    dict["CC"] = ("Correlation Coefficient",nx.degree_pearson_correlation_coefficient(G))
    e9 = time.time()
    print("CC:{0},total:{1}".format(format(e9-e8,".2f"),format(e9-s,".2f")))

    # try:
    #     dict["DIA"] = ("Diameter",nx.diameter(G))
    # except:
    #     dict["DIA"] = ("Diameter","INF")
    # dict["B"] = ("Betweenness",nx.betweenness_centrality(G))

    with open(os.path.join(savepath, name + ".pkl"),
              'wb') as fo:  # 将数据写入pkl文件
        pickle.dump(dict, fo)

def add_betweenness(G,name,savepath):
    dict = pickle.load(open(os.path.join(savepath, name + ".pkl"), 'rb'))
    s = time.time()
    # try:
    #     dict["ECC"] = ("Eccentricity",nx.diameter(G))
    # except:
    #     dict["DIA"] = ("Diameter","INF")
    # e1 = time.time()
    # print("DIA:{0},total:{1}".format(format(e1-s,".2f"),format(e1-s,".2f")))
    if name == "road-usroads":
        k=500
    elif name == "road-roadNet-CA":
        k=100
    else:
        k=None
    print("K={0}".format(k))
    dict["B"] = ("Betweenness",nx.betweenness_centrality(G,k))
    e1 = time.time()
    print("B:{0}".format(format(e1-s,".2f")))

    with open(os.path.join(savepath, name + ".pkl"),
              'wb') as fo:  # 将数据写入pkl文件
        pickle.dump(dict, fo)
def show_basic_information(dict):
    # Number of nodes
    # Number of edges
    # Diameter
    # Correlation Coefficient
    for key in dict:
       if key in ["NN","NE","DIA","CC"]:
           print("{0}:{1}".format(dict[key][0],dict[key][1]))

    # Maximum degree
    # Average degree
    print("Maximum degree:{0}".format(len(dict["DH"][1])-1))
    print("Average degree:{0}".format(2*dict["NE"][1]/dict["NN"][1]))

    # Average clustering coefficient
    # Numer of triangles
    print("Average clustering coefficient:{0}".format(sum([dict["C"][1][k] for k in dict["C"][1]])/dict["NN"][1]))
    print("Numer of triangles:{0}".format(sum([dict["T"][1][node] for node in dict["T"][1]])/3))
def calc_betweenness(G,k):
    s = time.time()
    b = nx.betweenness_centrality(G,k)
    e1 = time.time()
    print(len(b))
    print(e1-s)
def plot_degree_distribution(dicts,savepath,names,flag="bar"):
    color = ["orange","yellow","green","deepskyblue"]

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
            ax1.bar([i + idx * width for i in range(len(p))], p, width=width, label=name,color = color[idx])
        elif flag == "log":
            ax1.set_yscale("log")
            ax1.bar([i + idx * width for i in range(len(p))], p, width=width, label=name,color = color[idx])

        # ax2.plot([i for i in range(len(p))],p,color = 'red',label=name)

    fig.tight_layout()
    plt.title("Digree Distribution")
    plt.legend()
    plt.savefig(os.path.join(savepath,"_".join(names)+"_digree_distribution_"+flag+".png"))
    print("Finish plotting {0}".format("_".join(names)+"_digree_distribution_"+flag+".png"))
def plot_betweenness_distribution(dicts,savepath,names):
    color = ["rosybrown","orange","yellow","green","deepskyblue"]

    for idx,(name,dict) in enumerate(zip(names,dicts)):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("b")
        ax1.set_ylabel(r"$P_b$")
        betweenness_histogram=[]
        for key in dict["B"][1]:
            betweenness_histogram.append(dict["B"][1][key])
        betweenness_histogram.sort()
        betweenness_histogram = np.array(betweenness_histogram)
        betweenness_histogram = betweenness_histogram[betweenness_histogram>0]
        num = 50
        width = (np.log(betweenness_histogram[-1])-np.log(betweenness_histogram[0]))/num
        lim = [np.exp(width*i+np.log(betweenness_histogram[0]))for i in range(num+1)]
        for i in range(num):
            bet = betweenness_histogram[betweenness_histogram>lim[i]]
            bet = bet[bet<lim[i+1]]
            y = bet.shape[0]/dict["NN"][1]
            x = np.exp((np.log(lim[i])+np.log(lim[i+1]))/2)
            if i == 0:
                ax1.bar(x,y,width = (lim[i+1]-lim[i])*0.8,color=color[idx],label = name)
            else:
                ax1.bar(x,y,width = (lim[i+1]-lim[i])*0.8,color=color[idx])
        ax1.set_xscale("log")
        # ax1.set_yscale("log")
        plt.legend()
        fig.savefig(os.path.join(savepath, name + "_betweenness_distribution.png"))
        print("Finish plotting {0}".format(name+"_betweenness_distribution.png"))
def plot_degree_betweenness_fig(dicts,savepath,names):
    color = ["rosybrown","orange","yellow","green","deepskyblue"]

    for idx,(name,dict) in enumerate(zip(names,dicts)):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("k")
        ax1.set_ylabel("b")
        x=[]
        y=[]
        for key in dict["B"][1]:
            x.append(dict["D"][1][key])
            y.append(dict["B"][1][key])
        ax1.scatter(x,y,label = name,color = color[idx])
        # ax1.set_xscale("log")
        # ax1.set_yscale("log")
        plt.legend()
        fig.savefig(os.path.join(savepath, name + "_degree_betweenness.png"))
        print("Finish plotting {0}".format(name+"_degree_betweenness.png"))
def plot_clustering_distribution(dicts,savepath,names,flag = "bar"):
    color = ["rosybrown","orange","yellow","green","deepskyblue"]
    for idx,(name,dict) in enumerate(zip(names,dicts)):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("c")
        ax1.set_ylabel(r"$P_c$")
        clustering_histogram=[]
        for key in dict["C"][1]:
            clustering_histogram.append(dict["C"][1][key])
        clustering_histogram.sort()
        clustering_histogram = np.array(clustering_histogram)
        clustering_histogram = clustering_histogram[clustering_histogram>0]
        num = 100
        if flag == "log":
            width = (np.log(clustering_histogram[-1])-np.log(clustering_histogram[0]))/num
            lim = [np.exp(width*i+np.log(clustering_histogram[0]))for i in range(num+1)]
            for i in range(num):
                clu = clustering_histogram[clustering_histogram>lim[i]]
                clu = clu[clu<lim[i+1]]
                y = clu.shape[0]/dict["NN"][1]
                x = np.exp((np.log(lim[i])+np.log(lim[i+1]))/2)
                if i == 0:
                    ax1.bar(x,y,width = (lim[i+1]-lim[i])*0.8,color=color[idx],label = name)
                else:
                    ax1.bar(x,y,width = (lim[i+1]-lim[i])*0.8,color=color[idx])
            ax1.set_xscale("log")
            # ax1.set_yscale("log")
        elif flag == "bar":
            x=[]
            y=[]
            width = (clustering_histogram[-1] - clustering_histogram[0]) / num
            for i in range(num):
                clu = clustering_histogram[clustering_histogram > i*width]
                clu = clu[clu < (i+1)*width]
                y.append(clu.shape[0] / dict["NN"][1])
                x.append((i+0.5)*width)
            ax1.bar(x, y, width=width * 0.9, color=color[idx], label=name)
        else:
            raise ValueError("Check your flag!")
        plt.legend()
        fig.savefig(os.path.join(savepath, name + "_clustering_distribution_"+flag+".png"))
        print("Finish plotting {0}".format(name+"_clustering_distribution_"+flag+".png"))
def plot_degree_clustering_fig(dicts,savepath,names):
    color = ["rosybrown","orange","yellow","green","deepskyblue"]

    for idx,(name,dict) in enumerate(zip(names,dicts)):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("k")
        ax1.set_ylabel("b")
        x=[]
        y=[]
        for key in dict["C"][1]:
            x.append(dict["D"][1][key])
            y.append(dict["C"][1][key])
        ax1.scatter(x,y,label = name,color = color[idx])
        # ax1.set_xscale("log")
        # ax1.set_yscale("log")
        plt.legend()
        fig.savefig(os.path.join(savepath, name + "_degree_clustering.png"))
        print("Finish plotting {0}".format(name+"_degree_clustering.png"))
def save_graph(graph,savepath,name):
    #initialze Figure
    plt.figure(num=None, figsize=(500, 500), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.kamada_kawai_layout(graph)
    # pos = nx.spring_layout(graph)
    s = time.time()
    nx.draw_networkx_nodes(graph,pos,node_size=100000)
    e1 = time.time()
    print("Finish nodes drawing in {0}s, total {1}".format(e1-s,e1-s))

    nx.draw_networkx_edges(graph,pos,width=10)
    e2 = time.time()
    print("Finish edges drawing in {0}s, total {1}".format(e2-e1,e2-s))

    nx.draw_networkx_labels(graph,pos)
    e3 = time.time()
    print("Finish labels drawing in {0}s, total {1}".format(e3-e2,e3-s))

    cut = 1.05
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    xmin = cut * min(xx for xx, yy in pos.values())
    ymin = cut * min(yy for xx, yy in pos.values())
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.savefig(os.path.join(savepath,name+".pdf"),bbox_inches="tight")
    e4 = time.time()
    print("Finish saving in {0}s, total {1}".format(e4-e3,e4-s))

    pylab.close()
    del fig

def plot_degree_correlations(dicts,savepath,names):
    color = ["YlOrRd", "Oranges", "YlOrBr", "YlGn", "GnBu"]
    for idx, (name, dict) in enumerate(zip(names, dicts)):
        k_num = len(dict["DH"][1])-1
        labels = np.arange(k_num)+1
        e_num=len(dict["E"][1])
        print("E_num:{0}".format(e_num))
        fig, ax = plt.subplots(figsize=(100, 100), dpi=80)

        prob = np.zeros((k_num, k_num))
        for i, j in dict["E"][1]:
            prob[dict["D"][1][i] - 1][dict["D"][1][j] - 1] = prob[dict["D"][1][i] - 1][dict["D"][1][j] - 1] + 1
        prob = prob / len(dict["E"][1])
        im, cbar = heatmap(np.flip(prob,0), row_labels=np.flip(labels,0), col_labels=labels, ax=ax,
                           cmap=color[idx], cbarlabel="degree corrilation")
        fig.tight_layout()
        plt.show()
        # fig.savefig(os.path.join(savepath, name + "_degree_correlations.png"))
        # print("Finish plotting {0}".format(name+"_degree_correlations.png"))
