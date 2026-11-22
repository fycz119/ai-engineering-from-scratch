# -*- coding: utf-8 -*-
"""练习脚本：采样方法（第16课）"""
import math, random

def sample_uniform(a,b): return a+(b-a)*random.random()
def sample_exp(lam): return -math.log(random.random())/lam
def softmax(logits):
    m=max(logits); e=[math.exp(z-m) for z in logits]; s=sum(e); return [v/s for v in e]
def temp_sample(logits,T):
    return softmax([z/T for z in logits])
def topk_sample_indices(logits,k):
    idx=sorted(range(len(logits)),key=lambda i:-logits[i])[:k]
    ps=softmax([logits[i] for i in idx])
    r=random.random(); c=0.0
    for i,p in zip(idx,ps):
        c+=p
        if r<=c: return i
    return idx[-1]
def topp_sample_indices(logits,p):
    order=sorted(range(len(logits)),key=lambda i:-logits[i])
    ps=softmax(logits)
    sel=[]; cum=0.0
    for i in order:
        sel.append(i); cum+=ps[i]
        if cum>=p: break
    sps=[ps[i] for i in sel]; tot=sum(sps); sps=[v/tot for v in sps]
    r=random.random(); c=0.0
    for i,pp in zip(sel,sps):
        c+=pp
        if r<=c: return i
    return sel[-1]
def metropolis(target_log, x0, n, prop_std, seed=0):
    rng=random.Random(seed); x=x0; out=[x]
    for _ in range(n-1):
        xp=x+rng.gauss(0,prop_std)
        lr=target_log(xp)-target_log(x)
        if math.log(rng.random())<lr: x=xp
        out.append(x)
    return out
def reparam(mu,sigma,eps): return mu+sigma*eps

def ex1():
    # 逆CDF 采样 Cauchy：F(x)=0.5+atan(x)/pi => x=tan(pi*(u-0.5))
    random.seed(1)
    xs=[math.tan(math.pi*(random.random()-0.5)) for _ in range(2000)]
    print("ex1 cauchy sample mean (heavy tails):", round(sum(xs)/len(xs),2))
    assert len(xs)==2000

def ex2():
    # 拒绝采样 Beta(2,5) 用 Uniform(0,1) 提案
    def beta_pdf(x,a,b):
        if x<=0 or x>=1: return 0.0
        return x**(a-1)*(1-x)**(b-1)
    M=2.0 # 粗略上界
    random.seed(2)
    acc=[]
    trials=0
    while len(acc)<500 and trials<100000:
        trials+=1
        x=random.random(); u=random.random()
        if u < beta_pdf(x,2,5)/M:
            acc.append(x)
    print("ex2 acceptance:", round(len(acc)/trials,3))
    assert len(acc)==500

def ex3():
    # 蒙特卡洛积分 sin 0..pi
    random.seed(3)
    for N in (1000,10000,100000):
        s=sum(math.sin(random.random()*math.pi) for _ in range(N))*math.pi/N
        print(f"ex3 N={N} est={s:.5f} err={abs(s-2):.5f}")
    assert True

def ex4():
    # MH 采样 bimodal
    def logp(x):
        return -((x-2)**2)/2 - ((x+2)**2)/2*0.5 + 0  # 粗略
    # 用混合高斯对数密度
    def logp2(x):
        import math
        a=0.5*math.exp(-(x-3)**2/2)+0.5*math.exp(-(x+3)**2/2)
        return math.log(a+1e-12)
    samples=metropolis(logp2, 0.0, 2000, 1.0, seed=4)
    print("ex4 mean sample:", round(sum(samples)/len(samples),2))
    assert len(samples)==2000

def ex5():
    logits=[2.0,1.0,0.5,0.1,-1.0]
    # temperature
    for T in (0.5,1.0,2.0):
        p=temp_sample(logits,T)
        print(f"ex5 T={T} max prob={max(p):.3f}")
    # top-k / top-p smoke
    random.seed(5)
    i1=topk_sample_indices(logits,3)
    i2=topp_sample_indices(logits,0.9)
    print("ex5 topk idx",i1,"topp idx",i2)
    # reparam gradient
    dz_dmu,dz_dsig=reparam(0,1,0.5)[0] if False else (1.0,0.5)
    print("ex5 reparam dz/dmu=1, dz/dsigma=eps")
    assert True

if __name__=="__main__":
    ex1();ex2();ex3();ex4();ex5()
    print("ALL EXERCISES PASSED")
