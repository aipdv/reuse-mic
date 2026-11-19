import ctypes, sys, time; from pathlib import Path
import numpy as np
here = Path(__file__).parent
lib = ctypes.CDLL(str(here / "libreuse_engine.dylib"))
lib.re_create.restype = ctypes.c_void_p; lib.re_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
P = ctypes.POINTER(ctypes.c_float); lib.re_step.argtypes = [ctypes.c_void_p, P, P, P, P]
for spec in sys.argv[1:]:
    rate, layers = map(int, spec.split(":"))
    hop = round(160 * rate / 8000); hop += hop % 2; nb = hop + 1
    e = lib.re_create(str(here / "reuse_rt.bin").encode(), nb, layers)
    rng = np.random.default_rng(0)
    m = rng.random(nb, dtype=np.float32); p = (rng.random(nb, dtype=np.float32) - .5) * 6
    mo = np.zeros(nb, np.float32); po = np.zeros(nb, np.float32)
    a = [x.ctypes.data_as(P) for x in (m, p, mo, po)]
    for _ in range(20): lib.re_step(e, *a)
    w, c = [], []
    for _ in range(300):
        w0, c0 = time.perf_counter(), time.process_time(); lib.re_step(e, *a)
        w.append(time.perf_counter() - w0); c.append(time.process_time() - c0)
    w, c = np.array(w) * 1e3, np.array(c) * 1e3
    print(f"{rate:>5} Hz {layers:>2} layers: wall med {np.median(w):5.2f} p95 {np.percentile(w,95):5.2f} ms | cpu {np.median(c):5.2f} ms = {np.median(c)/20:.2f} cores (budget 20 ms)")
