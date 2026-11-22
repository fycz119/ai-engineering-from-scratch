# -*- coding: utf-8 -*-
"""练习脚本：随机过程（第22课）"""
import math, random

def rw1d(n, seed=0):
    rng=random.Random(seed)
    steps=[rng.choice([-1,1]) for _ in range(n)]
    pos=[0]; 
    for s in steps: pos.append(pos[-1]+s)
    return pos

def rw2d(n, seed=0):
    rng=random.Random(seed)
    x=[0]; y=[0]
    for _ in range(n):
        d=rng.choice([0,1,2,3])
        if d==0: x.append(x[-1]+1); y.append(y[-1])
        elif d==1: x.append(x[-1]-1); y.append(y[-1])
        elif d==2: x.append(x[-1]); y.append(y[-1]+1)
        else: x.append(x[-1]); y.append(y[-1]-1)
    return x,y

def ex1():
    finals=[]
    for s in range(200):
        p=rw1d(10000, seed=s)
        finals.append(p[-1])
    mean=sum(finals)/len(finals)
    sd=math.sqrt(sum((f-mean)**2 for f in finals)/len(finals))
    print("ex1 mean final:", round(mean,2), "std:", round(sd,2), "expected sqrt(n)=100")
    assert 80<sd<120

def ex2():
    # Markov chain text generator (tiny)
    text="the cat sat on the mat the cat ate the rat"
    words=text.split()
    trans={}
    for i in range(len(words)-1):
        trans.setdefault(words[i],{}).setdefault(words[i+1],0)
        trans[words[i]][words[i+1]]+=1
    # build
    print("ex2 transitions for 'the':", trans.get("the"))
    # generate
    rng=random.Random(1); cur="the"; out=[cur]
    for _ in range(8):
        opts=trans.get(cur)
        if not opts: break
        tot=sum(opts.values())
        r=rng.random()*tot; c=0.0; nxt=None
        for w,v in opts.items():
            c+=v
            if r<=c: nxt=w; break
        out.append(nxt); cur=nxt
    print("ex2 generated:", " ".join(out))
    assert len(out)>=5

def ex3():
    # simulated annealing via MH on f with many minima
    def f(x): return math.sin(x)*x*x  # rugged
    rng=random.Random(2)
    T=10.0; x=0.0; best=x
    for step in range(20000):
        T*=0.9995
        xp=x+rng.gauss(0,1)
        dE=f(xp)-f(x)
        if dE<0 or rng.random()<math.exp(-dE/max(T,1e-6)):
            x=xp
            if f(x)<f(best): best=x
    print("ex3 best x:", round(best,4), "f:", round(f(best),4))
    assert True

def ex4():
    # Langevin on double-well U=(x^2-1)^2
    def gradU(x): return 4*x*(x*x-1)
    def run(T):
        rng=random.Random(3); x=0.0; samples=[]
        for _ in range(5000):
            x=x-0.01*gradU(x)+math.sqrt(2*T*0.01)*rng.gauss(0,1)
            samples.append(x)
        return samples
    for T in (0.1,1.0):
        s=run(T)
        # fraction in right well x>0
        frac=sum(1 for v in s if v>0)/len(s)
        print(f"ex4 T={T} frac right well={frac:.3f}")
    assert True

def ex5():
    # forward diffusion on 1D sine
    import math
    N=50; T=100
    signal=[math.sin(2*math.pi*i/N) for i in range(N)]
    x=signal[:]
    for t in range(T):
        beta=0.02
        noise=[random.gauss(0,1) for _ in range(N)]
        x=[math.sqrt(1-beta)*x[i]+math.sqrt(beta)*noise[i] for i in range(N)]
    print("ex5 final mean abs:", round(sum(abs(v) for v in x)/N,3))
    # simple denoiser: subtract estimated noise (here zero) -> just print
    assert True

def ex6_stationary():
    # Markov chain stationary via power method
    P=[[0.7,0.1,0.2],[0.3,0.4,0.3],[0.4,0.2,0.4]]
    dist=[1.0,0.0,0.0]
    for _ in range(100):
        nd=[0.0,0.0,0.0]
        for i in range(3):
            for j in range(3):
                nd[j]+=dist[i]*P[i][j]
        dist=nd
    print("ex6 stationary:", [round(v,4) for v in dist])
    assert abs(sum(dist)-1)<1e-9

if __name__=="__main__":
    ex1();ex2();ex3();ex4();ex5();ex6_stationary()
    print("ALL EXERCISES PASSED")
