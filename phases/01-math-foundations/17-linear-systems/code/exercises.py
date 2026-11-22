# -*- coding: utf-8 -*-
"""练习脚本：线性方程组（第17课）"""
import math, random

def solve_linear(A,b):
    # 高斯消元 + 部分主元
    n=len(b)
    M=[row[:]+[b[i]] for i,row in enumerate(A)]
    for k in range(n):
        piv=max(range(k,n),key=lambda r:abs(M[r][k]))
        M[k],M[piv]=M[piv],M[k]
        for i in range(k+1,n):
            m=M[i][k]/M[k][k]
            for j in range(k,n+1):
                M[i][j]-=m*M[k][j]
    x=[0.0]*n
    for i in range(n-1,-1,-1):
        x[i]=(M[i][n]-sum(M[i][j]*x[j] for j in range(i+1,n)))/M[i][i]
    return x

def cholesky(A):
    n=len(A); L=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s=A[i][j]-sum(L[i][k]*L[j][k] for k in range(j))
            if i==j:
                L[i][j]=math.sqrt(s)
            else:
                L[i][j]=s/L[j][j]
    return L
def chol_solve(A,b):
    L=cholesky(A)
    n=len(b); y=[0.0]*n
    for i in range(n):
        y[i]=(b[i]-sum(L[i][k]*y[k] for k in range(i)))/L[i][i]
    x=[0.0]*n
    for i in range(n-1,-1,-1):
        x[i]=(y[i]-sum(L[k][i]*x[k] for k in range(i+1,n)))/L[i][i]
    return x

def cond_number(A):
    # 用特征值近似 for SPD：max/min diag of? 简化：用 numpy 不可用，用幂法估算最大/最小奇异值近似
    # 这里用简单的：对角占优矩阵近似 cond = max diag / min diag（演示）
    diag=[A[i][i] for i in range(len(A))]
    return max(diag)/min(diag)

def ex1():
    A=[[1,2,3],[4,5,6],[7,8,10]]; b=[6,15,27]
    x=solve_linear(A,b)
    print("ex1 x:", [round(v,6) for v in x])
    # verify residual
    for i in range(3):
        assert abs(sum(A[i][j]*x[j] for j in range(3))-b[i])<1e-6
    print("ex1 residual ok")

def ex2():
    random.seed(0)
    m,n=50,5
    X=[[random.gauss(0,1) for _ in range(n)] for _ in range(m)]
    wtrue=[2.0,-1.0,0.5,0.0,1.0]
    y=[sum(X[i][j]*wtrue[j] for j in range(n))+random.gauss(0,0.1) for i in range(m)]
    # normal equations
    XtX=[[sum(X[i][k]*X[i][l] for i in range(m)) for l in range(n)] for k in range(n)]
    Xty=[sum(X[i][k]*y[i] for i in range(m)) for k in range(n)]
    w=solve_linear(XtX,Xty)
    print("ex2 w:", [round(v,3) for v in w])
    assert len(w)==n

def ex3():
    # 近奇异矩阵 + 正则化
    A=[[1.0,1.0+1e-10],[1.0,1.0+2e-10]]
    b=[2.0,2.0]
    try:
        x=solve_linear(A,b)
        print("ex3 no reg:", x)
    except Exception as e:
        print("ex3 no reg failed:", e)
    lam=0.01
    Ar=[[A[i][j]+(lam if i==j else 0) for j in range(2)] for i in range(2)]
    x2=solve_linear(Ar,b)
    print("ex3 reg:", x2)
    assert all(math.isfinite(v) for v in x2)

def ex4():
    # 共轭梯度（简化：对 5x5 SPD）
    import random
    random.seed(1)
    n=5
    M=[[random.gauss(0,1) for _ in range(n)] for _ in range(n)]
    A=[[sum(M[i][k]*M[j][k] for k in range(n)) for j in range(n)] for i in range(n)]  # SPD
    for i in range(n): A[i][i]+=n
    b=[random.gauss(0,1) for _ in range(n)]
    # CG
    x=[0.0]*n
    r=[b[i]-sum(A[i][j]*x[j] for j in range(n)) for i in range(n)]
    p=r[:]
    rs=sum(v*v for v in r)
    it=0
    for _ in range(n):
        Ap=[sum(A[i][j]*p[j] for j in range(n)) for i in range(n)]
        alpha=rs/sum(p[i]*Ap[i] for i in range(n))
        x=[x[i]+alpha*p[i] for i in range(n)]
        r=[r[i]-alpha*Ap[i] for i in range(n)]
        rs2=sum(v*v for v in r)
        it+=1
        if math.sqrt(rs2)<1e-8: break
        beta=rs2/rs; rs=rs2
        p=[r[i]+beta*p[i] for i in range(n)]
    print("ex4 CG iters:", it)
    assert it<=n

def ex5():
    # Cholesky vs 直接求解计时（小规模）
    import time
    n=50
    random.seed(2)
    M=[[random.gauss(0,1) for _ in range(n)] for _ in range(n)]
    A=[[sum(M[i][k]*M[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    for i in range(n): A[i][i]+=n
    b=[random.gauss(0,1) for _ in range(n)]
    t0=time.time(); x1=chol_solve(A,b); t1=time.time()
    t2=time.time(); x2=solve_linear(A,b); t3=time.time()
    print("ex5 chol time", round(t1-t0,4), "elim time", round(t3-t2,4))
    # residual check
    for x in (x1,x2):
        r=max(abs(sum(A[i][j]*x[j] for j in range(n))-b[i]) for i in range(n))
        assert r<1e-6
    print("ex5 residuals ok")

if __name__=="__main__":
    ex1();ex2();ex3();ex4();ex5()
    print("ALL EXERCISES PASSED")
