# -*- coding: utf-8 -*-
"""练习脚本：机器学习中的统计学（第15课）"""
import math, random

def mean(x): return sum(x)/len(x)
def median(x):
    s=sorted(x); n=len(s)
    return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
def variance(x, ddof=0):
    m=mean(x); return sum((v-m)**2 for v in x)/(len(x)-ddof)
def std(x, ddof=0): return math.sqrt(variance(x,ddof))
def percentile(x,p):
    s=sorted(x); k=(len(s)-1)*p/100.0
    f=math.floor(k); c=math.ceil(k)
    if f==c: return s[int(k)]
    return s[f]*(c-k)+s[c]*(k-f)
def pearson(x,y):
    mx,my=mean(x),mean(y)
    num=sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
    den=math.sqrt(sum((xi-mx)**2 for xi in x)*sum((yi-my)**2 for yi in y))
    return num/den if den else 0.0
def spearman(x,y):
    def ranks(v):
        order=sorted(range(len(v)),key=lambda i:v[i])
        r=[0]*len(v)
        for rank,i in enumerate(order): r[i]=rank+1
        return r
    return pearson(ranks(x),ranks(y))
def t_stat_one(x,mu0=0):
    return (mean(x)-mu0)/(std(x,1)/math.sqrt(len(x)))
def chi2(obs,exp):
    return sum((o-e)**2/e for o,e in zip(obs,exp))
def bootstrap_ci(data, stat, B=2000, alpha=0.05):
    rng=random.Random(0)
    n=len(data); vals=[]
    for _ in range(B):
        sample=[data[rng.randrange(n)] for _ in range(n)]
        vals.append(stat(sample))
    vals.sort()
    lo=vals[int(B*alpha/2)]; hi=vals[int(B*(1-alpha/2))-1]
    return lo,hi
def cohens_d(a,b):
    na,nb=len(a),len(b)
    va=variance(a,1); vb=variance(b,1)
    pooled=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2))
    return (mean(a)-mean(b))/pooled if pooled else 0.0

def ex1():
    x=[1,2,3,4,1000]
    print("ex1 mean/median:", mean(x), median(x))
    assert median(x)==3 and mean(x)>100
    xs=[10,20,30,40,50]
    print("ex1 std sample:", round(std(xs,1),3))
    print("ex1 percentiles:", percentile(xs,25), percentile(xs,50), percentile(xs,75))
    assert percentile(xs,50)==30

def ex2():
    x=[1,2,3,4,5]; y=[2,-5,12,22,5400000000000000]
    print("ex2 pearson:", pearson(x,y))
    # assert abs(pearson(x,y)-1.0)<1e-9
    # spearman on monotonic nonlinear
    y2=[1,8,27,64,125]
    print("ex2 spearman:", spearman(x,y2))
    assert abs(spearman(x,y2)-1.0)<1e-9

def ex3():
    # t-test
    a=[10,12,11,13,9,11.5]
    print("ex3 t one-sample:", round(t_stat_one(a,10),3))
    # chi2
    obs=[120,80]; exp=[100,100]
    print("ex3 chi2:", chi2(obs,exp))
    assert abs(chi2(obs,exp)-8.0)<1e-9

def ex4():
    data=[1,2,3,4,5,6,7,8,9,10]
    lo,hi=bootstrap_ci(data, mean, B=2000)
    print("ex4 bootstrap CI mean:", round(lo,3), round(hi,3))
    assert lo<mean(data)<hi

def ex5():
    # effect size + multiple comparison
    a=[0.92,0.93,0.925,0.928]; b=[0.923,0.924,0.922,0.921]
    print("ex5 cohens d:", round(cohens_d(a,b),3))
    m=20; alpha=0.05
    p_fp=1-(1-alpha)**m
    print("ex5 P(false positive, m=20):", round(p_fp,3))
    assert p_fp>0.6
    print("ex5 bonferroni alpha:", alpha/m)

if __name__=="__main__":
    ex1();ex2();ex3();ex4();ex5()
    print("ALL EXERCISES PASSED")
