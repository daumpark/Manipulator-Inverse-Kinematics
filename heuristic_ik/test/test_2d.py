# heuristic_ik/scripts/test_2d.py
import numpy as np, time
from heuristic_ik.solvers import CCD2D, FABRIK2D

def bench_2d():
    L = [0.3, 0.25, 0.2, 0.15]
    CCD = CCD2D(L); FAB = FABRIK2D(L)
    q0 = np.zeros(len(L))
    for seed in range(5):
        np.random.seed(seed)
        tgt = np.random.uniform(low=[-0.6,-0.6], high=[0.6,0.6])
        for name, solver in [('CCD2D', CCD), ('FABRIK2D', FAB)]:
            t0=time.perf_counter()
            q, ok, info = solver.solve(tgt, q0)
            dt=(time.perf_counter()-t0)*1000.0
            print(f"[{name}] target={tgt}, t={dt:.2f} ms, iters={info['iters_total']}, pos_err={info['pos_err']:.4f}, ok={ok}")
        print('-'*60)

if __name__=='__main__':
    bench_2d()
