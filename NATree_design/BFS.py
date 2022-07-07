import numpy as np
import queue


def findPaths(adjtab, start, path=[]):
    path = path + [start]
    nodes = adjtab.get(start, [])
    if len(nodes) == 0:
        return [path]

    paths = []
    for node in nodes:
        if node not in path:
            newpaths = findPaths(adjtab, node, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def BFS_org(adjmat, start):
    visited = []
    output = []
    q = queue.Queue()
    q.put(start)
    visited.append(start)
    visited.append(-1)
    while not q.empty():
        u = q.get()
        output.append(u)
        for v in list(np.where(adjmat[u][:] == 1)[0]):
            if v not in visited:
                visited.append(v)
                q.put(v)
        visited.append(-1)

    return visited


def splitTree(list_ori, p):
    list_new = []
    list_short = []
    for i in list_ori:
        if i != p:
            list_short.append(i)
        else:
            list_new.append(list_short)
            list_short = []
    list_new.append(list_short)
    return list_new


def Addweight4Tree(X, weighted_file):
    X1 = X.astype(np.float32)
    for k in range(X.shape[0]):
        for row in range(1, X.shape[1]):
            for column in range(X.shape[2]):
                adj_x = X[k, row - 1, column]
                adj_y = X[k, row, column]
                # 替换对应矩阵上的值
                if weighted_file[int(adj_x), int(adj_y)] == 0:
                    X1[k, row, column] = 1
                else:
                    X1[k, row, column] = 1.0 / weighted_file[int(adj_x), int(adj_y)]

    X1[:, 0, :] = 1
    return X1
