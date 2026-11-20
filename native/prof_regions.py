"""Per-region profile of one hop: python prof_regions.py NB LAYERS THREADS (uses libreuse_prof.dylib)."""
import ctypes, sys, time; from pathlib import Path
import numpy as np
h = Path(__file__).parent; lib = ctypes.CDLL(str(h / "libreuse_prof.dylib"))
lib.re_create.restype = ctypes.c_void_p; lib.re_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
P = ctypes.POINTER(ctypes.c_float); lib.re_step.argtypes = [ctypes.c_void_p, P, P, P, P]
lib.re_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]; lib.re_rt_self.argtypes = [ctypes.c_double, ctypes.c_double]
lib.re_set_realtime.argtypes = [ctypes.c_double, ctypes.c_double]
nb, L, T = (int(a) for a in (sys.argv[1:4] + ["961", "4", "4"][len(sys.argv) - 1:])[:3])
ctypes.CDLL(None).pthread_set_qos_class_self_np(0x21, 0)  # user-interactive, not real-time: hops run back to back here
e = lib.re_create(str(h / "reuse_rt.bin").encode(), nb, L); lib.re_set_threads(e, T)
rng = np.random.default_rng(0); m = rng.random(nb, dtype=np.float32); p = m.copy(); mo = m.copy(); po = m.copy()
a = [x.ctypes.data_as(P) for x in (m, p, mo, po)]
wall = (ctypes.c_double * 64)(); work = (ctypes.c_double * 64)(); n = ctypes.c_int()
for _ in range(20): lib.re_step(e, *a)
lib.re_prof_regions(wall, work, ctypes.byref(n)); N = 200; t = time.perf_counter()
for _ in range(N): lib.re_step(e, *a)
tot = (time.perf_counter() - t) / N * 1e3; lib.re_prof_regions(wall, work, ctypes.byref(n))
W = sum(wall[i] for i in range(n.value)) / N
print(f"hop {tot:.2f} ms, in regions {W:.2f} ms, outside {tot - W:.2f} ms; threads {T}")
for i in range(n.value): print(f"  r{i:02d} wall {wall[i]/N:6.3f}  work {work[i]/N:6.3f}  speedup {work[i]/max(wall[i],1e-9):4.2f}")
