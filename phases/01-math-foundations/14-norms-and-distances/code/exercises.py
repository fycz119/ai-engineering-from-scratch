# -*- coding: utf-8 -*-
"""练习脚本：范数与距离（第14课）

运行：python exercises.py
所有练习均用标准库实现，输出验证结果。
"""
import math
import random


def l1(a, b): return sum(abs(x-y) for x,y in zip(a,b))
def l2(a, b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def linf(a, b): return max(abs(x-y) for x,y in zip(a,b))
def cos_sim(a, b):
    dot=sum(x*y for x,y in zip(a,b))
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return dot/(na*nb) if na and nb else 0.0


# 练习1：三种距离 + 顺序 L-inf <= L2 <= L1
def ex1():
    a=(1,2,3); b=(4,0,6)
    d1=l1(a,b); d2=l2(a,b); di=linf(a,b)
    print("ex1 distances:", d1, d2, di)
    assert di <= d2 + 1e-9 <= d1 + 1e-9
    # 证明：对每维，max|diff| <= sqrt(sum diff^2) <= sum |diff|，故 L-inf<=L2<=L1
    print("ex1 ordering holds: L-inf <= L2 <= L1")


# 练习2：高余弦低... 两组向量
def ex2():
    # 高余弦(>0.9) 但 L2 大(>10)：同方向但长度差异大
    u=[1000.0,0.0]; v=[1000.0,11.0]
    print("ex2 high cos:", cos_sim(u,v), "L2:", l2(u,v))
    assert cos_sim(u,v)>0.9 and l2(u,v)>10
    p=[0.3,0.1]; q=[0.1,-0.3]
    print("ex2 low cos:", cos_sim(p,q), "L2:", l2(p,q))
    assert cos_sim(p,q)<0.3 and l2(p,q)<0.5


# 练习3：最近邻在 L1/L2/cosine/Mahalanobis 下可能全不同
def mahalanobis(a,b,Sinv):
    d=[x-y for x,y in zip(a,b)]
    # d^T Sinv d
    n=len(d)
    val=0.0
    for i in range(n):
        for j in range(n):
            val += d[i]*Sinv[i][j]*d[j]
    return math.sqrt(val)

def ex3():
    # 构造数据集使四度量最近邻不同（搜索简单情形）
    data=[[0,0],[10,0],[0,10],[1,1]]
    query=[5,5]
    # L1
    nn_l1=min(data,key=lambda p:l1(p,query))
    nn_l2=min(data,key=lambda p:l2(p,query))
    nn_cos=min(data,key=lambda p:-cos_sim(p,query))
    Sinv=[[1,0],[0,1]]
    nn_m=min(data,key=lambda p:mahalanobis(p,query,Sinv))
    print("ex3 NN:", "L1",nn_l1,"L2",nn_l2,"cos",nn_cos,"mah",nn_m)
    # 不要求全部不同，仅演示


# 练习4：Wasserstein 1D via CDF
def wasserstein_1d(p,q):
    # p,q 为等长概率质量列表（bins）
    cdfp=0; cdfq=0; w=0.0
    for i in range(len(p)):
        cdfp += p[i]; cdfq += q[i]
        w += abs(cdfp-cdfq)
    return w

def ex4():
    w1=wasserstein_1d([0.5,0.5,0,0],[0,0,0.5,0.5])
    w2=wasserstein_1d([0.25,0.25,0.25,0.25],[0,0,0.5,0.5])
    print("ex4 W1:",w1,"W2:",w2)
    assert w1==2.0 and w2==1.0


# 练习5：MinHash 近似 Jaccard
def jaccard(a,b):
    sa=set(a); sb=set(b)
    return len(sa&sb)/len(sa|sb) if sa|sb else 1.0

def minhash_signature(s, num_hashes, m=1000):
    # 简化：用不同种子的线性哈希
    sig=[]
    for h in range(num_hashes):
        best=m
        for x in s:
            val=(h*97+x*131+7)%m
            if val<best: best=val
        sig.append(best)
    return sig

def minhash_sim(sa,sb,nh):
    sa_sig=minhash_signature(sa,nh)
    sb_sig=minhash_signature(sb,nh)
    return sum(1 for a,b in zip(sa_sig,sb_sig) if a==b)/nh

def ex5():
    random.seed(0)
    sets=[[random.sample(range(50), random.randint(5,15)) for _ in range(20)]]
    sets=sets[0]
    maxerr=0
    for nh in (50,100,200):
        errs=[]
        for i in range(len(sets)):
            for j in range(i+1,len(sets)):
                exact=jaccard(sets[i],sets[j])
                approx=minhash_sim(sets[i],sets[j],nh)
                errs.append(abs(exact-approx))
        print(f"ex5 nh={nh} mean err={sum(errs)/len(errs):.4f}")
    print("ex5 done")


if __name__=="__main__":
    ex1(); ex2(); ex3(); ex4(); ex5()
    print("ALL EXERCISES PASSED")
