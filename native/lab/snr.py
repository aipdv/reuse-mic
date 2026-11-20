"""SNR in dB of wav B against reference wav A (same length/rate): python snr.py A B"""
import sys, numpy as np, soundfile as sf
def rd(p): return sf.read(p, dtype="float64")[0]
a, b = rd(sys.argv[1]), rd(sys.argv[2]); n = min(len(a), len(b)); a, b = a[:n], b[:n]
print(f"{sys.argv[2]}: SNR {10*np.log10(np.sum(a*a)/max(np.sum((a-b)**2),1e-30)):.1f} dB, max|d| {np.abs(a-b).max():.2e}")
