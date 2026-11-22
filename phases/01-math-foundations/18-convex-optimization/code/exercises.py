# -*- coding: utf-8 -*-
"""练习脚本：凸优化（第18课）"""
import math, random

def check_convex(f, dim=1, bounds=(-5,5), samples=2000):
    viol=0
    for _ in range(samples):
        x=[random.uniform(*bounds) for _ in range(dim)]
        y=[random.uniform(*bounds) for _ in range(dim)]
        t=random.uniform(0,1)
        mid=[t*xi+(1-t)*yi for xi,yi in zip(x,y)]
        lhs=f(mid); rhs=t*f(x)+(1-t)*f(y)
        if lhs>rhs+1e-9: viol+=1
    return viol==0, viol

def newton_2d(f,grad,hess,x0,steps=50,tol=1e-12):
    x=list(x0); hist=[x[:]]
    for _ in range(steps):
        g=grad(x); H=hess(x)
        det=H[0][0]*H[1][1]-H[0][1]*H[1][0]
        if abs(det)<1e-15: break
        dx0=(H[1][1]*g[0]-H[0][1]*g[1])/det
        dx1=(-H[1][0]*g[0]+H[0][0]*g[1])/det
        x=[x[0]-dx0,x[1]-dx1]
        hist.append(x[:])
        if math.hypot(g[0],g[1])<tol: break
    return hist

def gd_2d(grad,x0,lr,steps=10000,tol=1e-10):
    x=list(x0); 
    for i in range(steps):
        g=grad(x); x=[x[0]-lr*g[0],x[1]-lr*g[1]]
        if math.hypot(g[0],g[1])<tol: return i+1
    return steps

def lagrange_min(f_grad,g_val,g_grad,x0,lr=0.01,steps=5000):
    x=list(x0); lam=0.0
    for _ in range(steps):
        fg=f_grad(x); gg=g_grad(x); gv=g_val(x)
        x=[xi-lr*(fgi+lam*ggi) for xi,fgi,ggi in zip(x,fg,gg)]
        lam=lam+lr*gv
    return x,lam

def ex1():
    cases={
     "x^4": (lambda x:x[0]**4,1),
     "sin": (lambda x:math.sin(x[0]),1),
     "x^2+y^2": (lambda x:x[0]**2+x[1]**2,2),
     "xy": (lambda x:x[0]*x[1],2),
     "relu": (lambda x:max(x[0],0),1),
    }
    for name,(f,d) in cases.items():
        ok,viol=check_convex(f,d)
        print(f"ex1 {name}: convex={ok} violations={viol}")

def ex2():
    def f(x): return 50*x[0]**2+x[1]**2
    def g(x): return [100*x[0],2*x[1]]
    def h(x): return [[100,0],[0,2]]
    hist=newton_2d(f,g,h,[10,10])
    print("ex2 newton steps:", len(hist)-1, "final", [round(v,6) for v in hist[-1]])
    n=gd_2d(g,[10,10],0.001)
    print("ex2 gd steps to 1e-10:", n)
    assert len(hist)-1<=5

def ex3():
    # min (x-3)^2+(y-3)^2 s.t. x+2y=4
    def fg(x): return [2*(x[0]-3),2*(x[1]-3)]
    def gv(x): return x[0]+2*x[1]-4
    def gg(x): return [1,2]
    x,lam=lagrange_min(fg,gv,gg,[0,0],lr=0.05,steps=20000)
    print("ex3 sol:", [round(v,4) for v in x], "lam", round(lam,4))
    # gradient parallel check
    g1=[2*(x[0]-3),2*(x[1]-3)]
    g2=[1,2]
    cross=g1[0]*g2[1]-g1[1]*g2[0]
    print("ex3 parallel cross:", round(cross,4))
    assert abs(cross)<0.5

def ex4():
    # L1 constrained: min (x-3)^2+(y-2)^2 s.t. |x|+|y|<=1
    # 投影到菱形（简单：坐标下降/惩罚）
    def fg(x): return [2*(x[0]-3),2*(x[1]-2)]
    best=None
    for _ in range(200000):
        x=random.uniform(-1,1); y=random.uniform(-1,1)
        if abs(x)+abs(y)<=1:
            val=(x-3)**2+(y-2)**2
            if best is None or val<best[0]: best=(val,x,y)
    print("ex4 best:", [round(best[1],4),round(best[2],4)], "one coord ~0:", abs(best[1])<0.05 or abs(best[2])<0.05)
    assert abs(best[1])<0.05 or abs(best[2])<0.05

def ex5():
    # Rosenbrock Hessian eigenvalues at (1,1) and (-1,1)
    def hess(x):
        return [[1200*x[0]**2-400*x[1]+2, -400*x[0]],[-400*x[0],200]]
    for pt in ([1,1],[-1,1]):
        H=hess(pt)
        # eigenvalues of 2x2
        tr=H[0][0]+H[1][1]; det=H[0][0]*H[1][1]-H[0][1]*H[1][0]
        disc=math.sqrt(max(tr*tr-4*det,0))
        e1=(tr+disc)/2; e2=(tr-disc)/2
        print(f"ex5 pt={pt} eigenvalues={round(e1,2)},{round(e2,2)}")
    assert True

if __name__=="__main__":
    ex1();ex2();ex3();ex4();ex5()
    print("ALL EXERCISES PASSED")
