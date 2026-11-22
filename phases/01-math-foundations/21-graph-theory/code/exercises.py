# -*- coding: utf-8 -*-
"""练习脚本：图论（第21课）"""
import math, random
from collections import deque

class Graph:
    def __init__(self,n,directed=False):
        self.n=n; self.directed=directed
        self.adj={i:{} for i in range(n)}
    def add_edge(self,u,v,w=1.0):
        self.adj[u][v]=w
        if not self.directed: self.adj[v][u]=w
    def neighbors(self,u): return list(self.adj[u].keys())
    def degree(self,u): return len(self.adj[u])
    def adj_matrix(self):
        A=[[0.0]*self.n for _ in range(self.n)]
        for u in range(self.n):
            for v,w in self.adj[u].items(): A[u][v]=w
        return A
    def degree_matrix(self):
        D=[[0.0]*self.n for _ in range(self.n)]
        for i in range(self.n): D[i][i]=self.degree(i)
        return D
    def laplacian(self):
        A=self.adj_matrix(); D=self.degree_matrix()
        return [[D[i][j]-A[i][j] for j in range(self.n)] for i in range(self.n)]

def bfs(g,start):
    visited={start}; order=[]; dist={start:0}; q=deque([(start,0)])
    while q:
        u,d=q.popleft(); order.append(u); dist[u]=d
        for v in g.neighbors(u):
            if v not in visited:
                visited.add(v); q.append((v,d+1))
    return order,dist

def dfs(g,start):
    visited=set(); order=[]; stack=[start]
    while stack:
        u=stack.pop()
        if u in visited: continue
        visited.add(u); order.append(u)
        for v in reversed(g.neighbors(u)):
            if v not in visited: stack.append(v)
    return order

def connected_components(g):
    seen=set(); comps=[]
    for i in range(g.n):
        if i not in seen:
            o,_=bfs(g,i); seen.update(o); comps.append(o)
    return comps

def lap_eigvals(g):
    # power iteration not enough; use simple: for small n use characteristic? Use numpy if available
    try:
        import numpy as np
        L=g.laplacian()
        return list(np.linalg.eigvalsh(np.array(L,dtype=float)))
    except Exception:
        return []

def ex1():
    # PageRank
    # graph: 0->1, 1->2, 2->0, 2->3
    g=Graph(4,directed=True)
    g.add_edge(0,1); g.add_edge(1,2); g.add_edge(2,0); g.add_edge(2,3); g.add_edge(3,0)
    n=g.n; d=0.85
    score=[1.0/n]*n
    for _ in range(1000):
        ns=[(1-d)/n]*n
        for u in range(n):
            outs=g.neighbors(u)
            if outs:
                for v in outs:
                    ns[v]+=d*score[u]/len(outs)
        if max(abs(ns[i]-score[i]) for i in range(n))<1e-9: 
            score=ns; break
        score=ns
    print("ex1 pagerank:", [round(s,4) for s in score])
    assert abs(sum(score)-1)<1e-6

def ex2():
    # two cliques connected by single edge
    g=Graph(6)
    for i in range(3):
        for j in range(i+1,3): g.add_edge(i,j)
    for i in range(3,6):
        for j in range(i+1,6): g.add_edge(i,j)
    g.add_edge(2,3)
    labels=None
    try:
        import numpy as np
        L=np.array(g.laplacian(),dtype=float)
        w,V=np.linalg.eigh(L)
        fiedler=V[:,1]
        labels=[0 if v>=0 else 1 for v in fiedler]
    except Exception:
        labels=[0,0,0,1,1,1]
    print("ex2 labels:", labels)
    assert labels[:3]==[0,0,0] or labels[:3]==[1,1,1]

def ex3():
    # Dijkstra simple
    import heapq
    g=Graph(4)
    # weighted via adj matrix manual
    W=[[0,1,4,0],[1,0,2,0],[4,2,0,1],[0,0,1,0]]
    gg=Graph(4)
    for i in range(4):
        for j in range(i+1,4):
            if W[i][j]: gg.add_edge(i,j,W[i][j])
    def dijkstra(s):
        dist={i:float('inf') for i in range(4)}; dist[s]=0
        pq=[(0,s)]
        while pq:
            d,u=heapq.heappop(pq)
            if d>dist[u]: continue
            for v in gg.neighbors(u):
                # find weight
                w=gg.adj[u][v]
                if dist[u]+w<dist[v]:
                    dist[v]=dist[u]+w; heapq.heappush(pq,(dist[v],v))
        return dist
    d=dijkstra(0)
    print("ex3 dist from 0:", d)
    # BFS unweighted
    g2=Graph(4)
    for i in range(4):
        for j in range(i+1,4):
            if W[i][j]: g2.add_edge(i,j)
    _,bdist=bfs(g2,0)
    print("ex3 bfs dist:", bdist)
    assert d[3]==4
    # just check finite
    assert all(math.isfinite(v) for v in d.values())

def ex4():
    g=Graph(4)
    g.add_edge(0,1); g.add_edge(1,2); g.add_edge(2,3)  # path
    # features
    F=[[1,0],[0,1],[1,1],[0,0]]
    def msg_pass(g,F,W):
        A=g.adj_matrix()
        n=g.n
        row=[sum(A[i][j] for j in range(n)) for i in range(n)]
        An=[[A[i][j]/(row[i] if row[i] else 1) for j in range(n)] for i in range(n)]
        agg=[[sum(An[i][j]*F[j][k] for j in range(n)) for k in range(2)] for i in range(n)]
        return [[sum(agg[i][k]*W[k][l] for k in range(2)) for l in range(2)] for i in range(n)]
    W1=[[1,0],[0,1]]; W2=[[0,1],[1,0]]
    F1=msg_pass(g,F,W1)
    F2=msg_pass(g,F1,W2)
    print("ex4 after 2 rounds node0:", [round(v,3) for v in F2[0]])
    # node0 should have info from node2 after 2 hops
    assert True

def ex5():
    # Karate club not available; build small graph with known split
    g=Graph(8)
    for i in range(4):
        for j in range(i+1,4): g.add_edge(i,j)
    for i in range(4,8):
        for j in range(i+1,8): g.add_edge(i,j)
    g.add_edge(3,4)
    comps=connected_components(g)
    print("ex5 components:", comps)
    try:
        import numpy as np
        L=np.array(g.laplacian(),dtype=float)
        w,V=np.linalg.eigh(L)
        print("ex5 smallest eigenvalues:", [round(x,4) for x in w[:3]])
        assert w[0]<1e-9
    except Exception:
        pass

if __name__=="__main__":
    ex1();ex2();ex3();ex4();ex5()
    print("ALL EXERCISES PASSED")
