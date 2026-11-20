import ctypes, sys, time; from pathlib import Path
import numpy as np
h = Path(__file__).parent; lib = ctypes.CDLL(str(h / "libreuse_prof.dylib"))
lib.re_create.restype = ctypes.c_void_p; lib.re_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
P = ctypes.POINTER(ctypes.c_float); lib.re_step.argtypes = [ctypes.c_void_p, P, P, P, P]
nb = int(sys.argv[1]) if len(sys.argv) > 1 else 241; L = int(sys.argv[2]) if len(sys.argv) > 2 else 4
e = lib.re_create(str(h / "reuse_rt.bin").encode(), nb, L)
m = np.random.rand(nb).astype(np.float32); p = m.copy(); mo = m.copy(); po = m.copy()
a = [x.ctypes.data_as(P) for x in (m, p, mo, po)]
for _ in range(20): lib.re_step(e, *a)
out = (ctypes.c_double * 8)(); lib.re_prof(out)
t = time.perf_counter(); N = 200
for _ in range(N): lib.re_step(e, *a)
tot = (time.perf_counter() - t) / N * 1e3; lib.re_prof(out)
names = {0: "ssm scan", 2: "sgemm (all)", 5: "conv+silu", 6: "ln+prelu"}  # softplus + gate run inside the scan pre-pass
print(f"total {tot:.2f} ms/hop; scan groups per hop {out[3]/N:.0f}, on the clamped slow path {out[1]/N:.0f}")
for i, n in names.items(): print(f"  {n:16s} {out[i]/N:6.2f} ms")
print(f"  {'other':16s} {tot - sum(out[i] for i in names)/N:6.2f} ms")
