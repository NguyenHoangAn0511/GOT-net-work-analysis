import streamlit as st
import pandas as pd
import networkx as nx
import community as community_louvain
import community.community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import cluster

def main():
    st.header('Game of Thorne Network analyst')
    MENU = ['Home page', 'Data analyst', 'Network building']
    choice = st.sidebar.selectbox('Navigation', MENU)
    if choice == MENU[0]:
        home()
    if choice == MENU[1]:
        DA()
    if choice == MENU[2]:
        NB()



def home():
    image = Image.open('data/ned.jpg')
    st.image(image, caption='Game of throne', use_column_width=True)
    st.header('Member:')
    st.markdown('- _**Nguyen Hoang An**_ - 18520430')
    st.markdown('- _**Duong Trong Van**_ - 18521630')


def DA():
    st.header('I - Data analyst')
    st.markdown('**_1.1 Edge checking_**')
    book1 = pd.read_csv('data/asoiaf-book1-edges.csv')
    book1.drop(['Type', 'book'],axis=1, inplace=True)
    st.dataframe(book1)
    r, c = book1.shape
    st.write(' - Number of edges: ', r)

    st.markdown('**1.2 Duplicate edges checking**')
    with st.echo():
        duplicate = book1.duplicated().sum()
    st.write('- Duplicate rows of diamonds DataFrame: ', duplicate)
    st.write('=> NO DUPLICATE. Let\'s check how many NODES in our network?')

    st.markdown('_**1.3 Nodes checking**_')
    nodeSource = book1['Source']
    nodeSource = nodeSource.rename({'Source': 'Node'})
    nodeTarget = book1['Target']
    nodeTarget = nodeTarget.rename({'Target': 'Node'})
    nodeChecking = nodeSource.append(nodeTarget)
    nodeChecking = nodeChecking.drop_duplicates()
    nodeChecking.reset_index(drop=True, inplace=True)
    r, = nodeChecking.shape
    st.write('NUMBER OF NODES: ', r)
    st.write('There are ', r ,' **NODES** in total of our networks. But any _**NULL VALUES**_ in our dataframe?')

    st.markdown('_**1.4 Checking null values**_')
    with st.echo():
        pd.isnull(book1).sum() > 0
    st.write('=> NO NULL in our dataframe. PERFECT!!!. Let\'s move to next part')

def NB():
    st.header('II - Network building')

    book1 = pd.read_csv('data/asoiaf-book1-edges.csv')
    book1.drop(['Type', 'book'],axis=1, inplace=True)
    nodeSource = book1['Source']
    nodeSource = nodeSource.rename({'Source': 'Node'})
    nodeTarget = book1['Target']
    nodeTarget = nodeTarget.rename({'Target': 'Node'})
    nodeChecking = nodeSource.append(nodeTarget)
    nodeChecking = nodeChecking.drop_duplicates()
    nodeChecking.reset_index(drop=True, inplace=True)

    Graph = nx.Graph()
    for _, edge in book1.iterrows():
        Graph.add_edge(edge['Source'], edge['Target'], weight=edge['weight'])
    
    st.markdown('_**2.1 Geometric measures**_')
    st.write('- Degree Centrality')
    st.write('- Closeness Centrality')
    
    st.markdown('> _**2.1.1 Degree Centrality**_')
    degreeCen= nx.degree_centrality(Graph)
    dfDegreeCen = pd.DataFrame(list(degreeCen.items()),columns = ['Character','Degree Centrality'])
    dfDegreeCen.sort_values("Degree Centrality", axis = 0, ascending = False, inplace = True)
    st.dataframe(dfDegreeCen)

    st.markdown('> _**2.1.2 Closeness Centrality**_')
    closenessCen= nx.closeness_centrality(Graph)
    dfClosenessCen = pd.DataFrame(list(closenessCen.items()),columns = ['Character','Closeness Centrality'])
    dfClosenessCen.sort_values("Closeness Centrality", axis = 0, ascending = False, inplace = True) 
    st.dataframe(dfClosenessCen)

    st.markdown('_**Spectral measures**_')
    st.write('- Page rank')
    st.write('- Eigenvector Centrality')

    st.markdown('> _**2.2.1 Page rank**_')
    pageRank =nx.pagerank(Graph, tol=0.001)
    dfPageRank = pd.DataFrame(list(pageRank.items()),columns = ['Character','Page Rank'])
    dfPageRank.sort_values("Page Rank", axis = 0, ascending = False, inplace = True) 
    st.dataframe(dfPageRank)

    st.markdown('> _**2.2.2 Eigenvector Centrality**_')
    eigenCen = nx.eigenvector_centrality(Graph)
    dfEigenCen = pd.DataFrame(list(eigenCen.items()),columns = ['Character','Eigenvector Centrality'])
    dfEigenCen.sort_values("Eigenvector Centrality", axis = 0, ascending = False, inplace = True)
    st.dataframe(dfEigenCen)

    st.markdown('_**2.3 Path-based measures**_')
    st.write('- Betweeness Centrality')
    betweenCen = nx.betweenness_centrality(Graph)
    dfBetweenCen = pd.DataFrame(list(betweenCen.items()),columns = ['Character','Betweenness Centrality'])
    dfBetweenCen.sort_values("Betweenness Centrality", axis = 0, ascending = False, inplace = True) 
    st.dataframe(dfBetweenCen)

    sum = pd.DataFrame((dfDegreeCen['Character'].values,
        dfClosenessCen['Character'].values,
        dfPageRank['Character'].values,
        dfEigenCen['Character'].values,
        dfBetweenCen['Character'].values),
        index=['Degree Centrality', 'Closeness Centrality', 'Pagerank', 'Eigenvector Centrality', 'Betweenness Centrality']).transpose()

    st.header('Let\'s **sumary** our _**Network Cetrality**_')
    st.dataframe(sum)

    st.markdown('_**2.4 Community detection**_')
    st.write('- Modularity Clustering')
    st.write('- K-mean Clustering')

    st.markdown('> _**2.4.1 Modularity Clustering**_')
    modulCluster = community.community_louvain.best_partition(Graph,weight='None')
    dfModulCluster = pd.DataFrame(list(modulCluster.items()),columns = ['Character','Community'])
    dfModulCluster.sort_values("Character",  inplace = True)
    st.dataframe(dfModulCluster)

    st.write('>> Number of characters in each community')
    modulCommunity = dfModulCluster["Community"].value_counts()
    modulCommunity = pd.DataFrame(list(modulCommunity.items()),columns = ['Community','Number of Character'])
    modulCommunity.sort_values("Community",  inplace = True)
    st.dataframe(modulCommunity)

    st.markdown('> _**2.4.2 K-mean Clustering**_')
    nodeChecking.reset_index(drop=True, inplace=True)
    data_dict = nodeChecking.to_dict()
    data_dict = {v: k for k, v in data_dict.items()}

    X = book1.copy()
    X.drop(['weight'],axis=1,inplace = True)
    X = X.replace({"Source": data_dict})
    X = X.replace({"Target": data_dict})

    G = nx.Graph()
    for _, edge in X.iterrows():
        G.add_edge(edge['Source'], edge['Target'])

    edge_mat = graph_to_edge_matrix(G)

    kmeans = cluster.KMeans(n_clusters=7).fit(edge_mat)
    resultKmeans = kmeans.labels_

    results = nodeChecking.to_dict()
    results = {v: k for k, v in results.items()}
    i = 0
    for key, value in results.items():
        results[key] = resultKmeans[i]
        i+=1

    dfKmeansCluster = pd.DataFrame(list(results.items()),columns = ['Character','Community'])
    dfKmeansCluster.sort_values("Character",  inplace = True)
    st.dataframe(dfKmeansCluster)

    kmeansCommunity = dfKmeansCluster["Community"].value_counts()
    kmeansCommunity = pd.DataFrame(list(kmeansCommunity.items()),columns = ['Community','Number of Character'])
    kmeansCommunity.sort_values("Community",  inplace = True)
    st.write('>> Number of characters in each community')
    st.dataframe(kmeansCommunity)


def graph_to_edge_matrix(G):
    edge_mat = np.zeros((len(G), len(G)), dtype=int)
    for node in G:
        for neighbor in G.neighbors(node):
            edge_mat[node][neighbor] = 1
        edge_mat[node][node] = 1
    return edge_mat

if __name__ == '__main__':
    main()