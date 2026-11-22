# -*- coding: utf-8 -*-
"""练习脚本：复数（第19课）"""
import math, random

class C:
    def __init__(self,r,i=0.0): self.r=r; self.i=i
    def __add__(self,o): return C(self.r+o.r,self.i+o.i)
    def __mul__(self,o): return C(self.r*o.r-self.i*o.i, self.r*o.i+self.i*o.r)
    def __truediv__(self,o):
        d=o.r*o.r+o.i*o.i
        return C((self.r*o.r+self.i*o.i)/d,(self.i*o.r-self.r*o.i)/d)
    def mag(self): return math.hypot(self.r,self.i)
    def phase(self): return math.atan2(self.i,self.r)
    def conj(self): return C(self.r,-self.i)

def euler(t): return C(math.cos(t),math.sin(t))
def to_polar(z): return z.mag(),z.phase()
def from_polar(r,t): return C(r*math.cos(t),r*math.sin(t))

def ex1():
    z=C(2,3)*C(4,-1)
    print("ex1 (2+3i)(4-i)=", z.r, z.i)
    assert abs(z.r-11)<1e-9 and abs(z.i-10)<1e-9
    w=C(5,2)/C(1,-3)
    print("ex1 (5+2i)/(1-3i)=", round(w.r,4), round(w.i,4))

def ex2():
    z=C(1,0)
    pts=[]
    for _ in range(12):
        z=z*euler(math.pi/6)
        pts.append((round(z.r,6),round(z.i,6)))
    print("ex2 after 12 steps:", pts[-1])
    assert abs(z.r-1)<1e-9 and abs(z.i)<1e-9

def ex3():
    # DFT
    def dft(sig):
        N=len(sig); out=[]
        for k in range(N):
            tot=C(0,0)
            for n in range(N):
                ang=-2*math.pi*k*n/N
                tot=tot+C(sig[n],0)*euler(ang)
            out.append(tot)
        return out
    N=32
    sig=[math.sin(2*math.pi*3*t/N)+0.5*math.sin(2*math.pi*7*t/N) for t in range(N)]
    sp=dft(sig)
    mags=[s.mag() for s in sp]
    # peaks at 3 and 7 (and symmetric)
    print("ex3 peak3 mag", round(mags[3],2), "peak7 mag", round(mags[7],2))
    assert mags[3]>mags[7]*1.5

def ex4():
    roots=[euler(2*math.pi*k/8) for k in range(8)]
    s=C(0,0)
    for r in roots: s=s+r
    print("ex4 sum roots:", round(s.r,6), round(s.i,6))
    assert abs(s.r)<1e-9 and abs(s.i)<1e-9
    # multiply by primitive root gives next
    w=euler(2*math.pi/8)
    assert abs((roots[0]*w).r-roots[1].r)<1e-9

def ex5():
    random.seed(0)
    maxdiff=0
    for _ in range(10):
        theta=random.uniform(-math.pi,math.pi)
        x=random.uniform(-5,5); y=random.uniform(-5,5)
        # complex mult
        z=C(x,y)*euler(theta)
        # matrix
        xr=x*math.cos(theta)-y*math.sin(theta)
        yr=x*math.sin(theta)+y*math.cos(theta)
        maxdiff=max(maxdiff,abs(z.r-xr),abs(z.i-yr))
    print("ex5 max diff:", maxdiff)
    assert maxdiff<1e-9

if __name__=="__main__":
    ex1();ex2();ex3();ex4();ex5()
    print("ALL EXERCISES PASSED")
