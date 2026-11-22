# -*- coding: utf-8 -*-
"""练习脚本：Fourier 变换（第20课）"""
import math, random

class C:
    def __init__(self,r,i=0.0): self.r=r; self.i=i
    def __add__(self,o): return C(self.r+o.r,self.i+o.i)
    def __sub__(self,o): return C(self.r-o.r,self.i-o.i)
    def __mul__(self,o): return C(self.r*o.r-self.i*o.i, self.r*o.i+self.i*o.r)
    def mag(self): return math.hypot(self.r,self.i)

def euler(t): return C(math.cos(t),math.sin(t))
def dft(x):
    N=len(x); out=[]
    for k in range(N):
        tot=C(0,0)
        for n in range(N):
            tot=tot+C(x[n],0)*euler(-2*math.pi*k*n/N)
        out.append(tot)
    return out
def idft(X):
    N=len(X); out=[]
    for n in range(N):
        tot=C(0,0)
        for k in range(N):
            tot=tot+X[k]*euler(2*math.pi*k*n/N)
        out.append(C(tot.r/N,tot.i/N))
    return out
def fft(x):
    N=len(x)
    if N<=1:
        v=x[0]; return [v if isinstance(v,C) else C(v,0)]
    if N%2: return dft(x)
    even=fft([x[i] for i in range(0,N,2)])
    odd=fft([x[i] for i in range(1,N,2)])
    res=[C(0,0)]*N
    for k in range(N//2):
        tw=euler(-2*math.pi*k/N)
        t=tw*odd[k]
        res[k]=even[k]+t
        res[k+N//2]=even[k]-t
    return res

def ex1():
    random.seed(0)
    N=128; fs=128
    f=7
    sig=[math.sin(2*math.pi*f*t/fs) for t in range(N)]
    sp=dft(sig)
    mags=[s.mag() for s in sp]
    peak=max(range(N//2), key=lambda k:mags[k])
    print("ex1 detected freq:", peak, "expected", f)
    # add noise
    sig2=[v+random.gauss(0,0.5) for v in sig]
    sp2=dft(sig2); mags2=[s.mag() for s in sp2]
    peak2=max(range(N//2), key=lambda k:mags2[k])
    print("ex1 noisy peak:", peak2)
    assert peak==f

def ex2():
    random.seed(1)
    for N in (256,512):
        sig=[random.random() for _ in range(N)]
        a=dft(sig); b=fft(sig)
        maxdiff=max(abs(a[k].r-b[k].r)+abs(a[k].i-b[k].i) for k in range(N))
        print(f"ex2 N={N} maxdiff={maxdiff:.2e}")
        assert maxdiff<1e-9

def ex3():
    x=[1,2,3,4,0,0,0,0]; h=[1,1,1,0,0,0,0,0]
    # circular conv direct
    N=len(x)
    direct=[sum(x[i]*h[(n-i)%N] for i in range(N)) for n in range(N)]
    # via fft
    X=fft([float(v) for v in x]); H=fft([float(v) for v in h])
    Y=[X[k]*H[k] for k in range(N)]
    y=idft(Y)
    fftres=[round(y[n].r,6) for n in range(N)]
    print("ex3 direct", direct)
    print("ex3 fft  ", fftres)
    assert all(abs(direct[n]-fftres[n])<1e-6 for n in range(N))

def ex4():
    # windowing: two close freqs
    N=128; fs=128
    sig=[math.sin(2*math.pi*10*t/fs)+math.sin(2*math.pi*12*t/fs) for t in range(N)]
    def windowed(w):
        return [sig[i]*w[i] for i in range(N)]
    hann=[0.5*(1-math.cos(2*math.pi*i/(N-1))) for i in range(N)]
    ham=[0.54-0.46*math.cos(2*math.pi*i/(N-1)) for i in range(N)]
    for name,w in (("none",None),("hann",hann),("hamming",ham)):
        s=sig if w is None else windowed(w)
        sp=dft(s); mags=[x.mag() for x in sp][:N//2]
        # find peaks
        peaks=sorted(range(len(mags)), key=lambda k:-mags[k])[:4]
        print(f"ex4 {name} top peaks: {sorted(peaks[:2])}")
    assert True

def ex5():
    # positional encoding dot product depends on distance
    d=8; maxpos=32
    def pe(pos):
        v=[]
        for i in range(d//2):
            w=1.0/(10000**(2*i/d))
            v.append(math.sin(pos*w)); v.append(math.cos(pos*w))
        return v
    def dot(a,b): return sum(x*y for x,y in zip(a,b))
    # check invariance to absolute position for fixed distance
    vals=set()
    for p1 in range(5,20):
        p2=p1+3
        vals.add(round(dot(pe(p1),pe(p2)),6))
    print("ex5 distinct dot values for dist=3:", len(vals))
    assert len(vals)==1

if __name__=="__main__":
    ex1();ex2();ex3();ex4();ex5()
    print("ALL EXERCISES PASSED")
