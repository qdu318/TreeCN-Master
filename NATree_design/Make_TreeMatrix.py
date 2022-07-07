from NATree_design.BFS import BFS_org, splitTree, findPaths, Addweight4Tree
import numpy as np
import pandas as pd


def calPathWeight(paths, weight, noDuplicated=True):
    index = {}
    for i in range(len(paths)):
        if noDuplicated:
            if len(set(paths[i])) < len(paths[i]):
                continue
        currentWeight = 0
        for j in range(1, len(paths[i])):
            currentWeight += weight[paths[i][j - 1]][paths[i][j]]
        tmp = {i: currentWeight}
        index.update(tmp)

    return index


def findPaddingPath(paths, weight, paddinglength):
    padingPaths = []
    index = calPathWeight(paths, weight, True)

    shortIndex = {}
    if len(index) < paddinglength:
        shortIndex = calPathWeight(paths, weight, False)

    index = sorted(index.items(), key=lambda x: x[1], reverse=True)
    shortIndex = sorted(shortIndex.items(), key=lambda x: x[1], reverse=True)
    pathIndex = []
    for i in range(len(index)):
        pathIndex.append(index[i][0])
    for j in range(len(shortIndex)):
        pathIndex.append(shortIndex[j][0])

    j = 0
    for i in range(paddinglength):
        padingPaths.append(paths[pathIndex[j]])
        j += 1
    return padingPaths


def graph2TreeMat(adjMat, distanceMat, savePath, save=True):
    graph = adjMat
    nodes_number = adjMat.shape[0]
    max_node_number = 0
    max_layer_number = 0
    max_node_child_number = 0

    for i in range(len(graph)):
        tmp_max_node_child_number = np.where(adjMat[0][:] == 1)[0].shape[0]
        if tmp_max_node_child_number > max_node_child_number:
            max_node_child_number = tmp_max_node_child_number

    for i in range(nodes_number):
        output = BFS_org(graph, i)
        splited = splitTree(output, -1)
        j = 1
        tempGraph = {}
        for k in range(len(output)):
            if output[k] != -1:
                tempGraph[output[k]] = splited[j]
                j += 1
        allpaths = []
        allpaths = findPaths(tempGraph, i, allpaths)

        for k in range(len(allpaths)):
            tmp_max_layer_number = len(allpaths[k])
            if tmp_max_layer_number > max_layer_number:
                max_layer_number = tmp_max_layer_number
        tmp_max_node_number = len(allpaths)
        if tmp_max_node_number > max_node_number:
            max_node_number = tmp_max_node_number

    y = np.zeros([1, max_layer_number, max_node_number * max_node_child_number])
    for i in range(nodes_number):
        output = BFS_org(graph, i)
        splited = splitTree(output, -1)
        j = 1
        tree = {}
        for k in range(len(output)):
            if output[k] != -1:
                tree[output[k]] = splited[j]
                j += 1

        paths = []
        paths = findPaths(tree, i, paths)

        for j in range(len(paths)):
            while len(paths[j]) < max_layer_number:
                paths[j].append(0)

        paddingSequence = [0] * max_layer_number
        while len(paths) < max_node_number * max_node_child_number:
            paths.append(paddingSequence)

        paths = np.array(paths).T

        if i == 0:
            y = np.reshape(paths, (1, max_layer_number, max_node_number * max_node_child_number))
        else:
            y = np.concatenate((y, np.reshape(paths,
                                              (1, max_layer_number, max_node_number * max_node_child_number))),
                               axis=0)

    print(y.shape)
    y_weight = Addweight4Tree(y, distanceMat)
    if save:
        np.save(savePath, y_weight)
    return y_weight


if __name__ == "__main__":
    # filepath = "../data_set/RandomUniformity/"
    filepath = "../data_set/SmallScaleAggregation/"
    distanceFile = "distance_50.csv"
    adjFile = "adjmat_50.csv"
    saveFile = "TreeMatrix_50.npy"
    A = pd.read_csv(filepath + adjFile, delimiter=',', header=None).values
    distanceMat = pd.read_csv(filepath + distanceFile, delimiter=',', header=None).values
    graph2TreeMat(A, distanceMat, filepath + saveFile, save=True)
