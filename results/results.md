# Results
Seed 7; full run. All numbers are from simulation on synthetic random patterns, apart from rows labelled theory.

## E1. One-step bit error rate: simulation vs Gaussian crosstalk theory

Probe = a stored pattern. We count bits whose single update would flip. The theory curve is Phi(-S/sqrt((P-1)V)), with the variance V computed exactly from the binomial law of the overlaps.

| n | N | P | P/N^(n-1) | sim bit err | theory |
|---|---|---|---|---|---|
| 2 | 256 | 5 | 0.020 | 0.0000 | 0.0000 |
| 2 | 256 | 9 | 0.035 | 0.0000 | 0.0000 |
| 2 | 256 | 14 | 0.055 | 0.0000 | 0.0000 |
| 2 | 256 | 24 | 0.094 | 0.0007 | 0.0004 |
| 2 | 256 | 40 | 0.156 | 0.0055 | 0.0053 |
| 2 | 256 | 66 | 0.258 | 0.0231 | 0.0238 |
| 2 | 256 | 110 | 0.430 | 0.0626 | 0.0631 |
| 2 | 256 | 184 | 0.719 | 0.1204 | 0.1189 |
| 2 | 256 | 307 | 1.199 | 0.1820 | 0.1807 |
| 3 | 64 | 82 | 0.020 | 0.0000 | 0.0000 |
| 3 | 64 | 137 | 0.033 | 0.0010 | 0.0009 |
| 3 | 64 | 228 | 0.056 | 0.0059 | 0.0077 |
| 3 | 64 | 380 | 0.093 | 0.0346 | 0.0304 |
| 3 | 64 | 635 | 0.155 | 0.0664 | 0.0736 |
| 3 | 64 | 1059 | 0.259 | 0.1243 | 0.1309 |
| 3 | 64 | 1766 | 0.431 | 0.2008 | 0.1924 |
| 3 | 64 | 2946 | 0.719 | 0.2604 | 0.2506 |
| 3 | 64 | 4915 | 1.200 | 0.3106 | 0.3013 |
| 4 | 32 | 655 | 0.020 | 0.0383 | 0.0367 |
| 4 | 32 | 1093 | 0.033 | 0.0847 | 0.0829 |
| 4 | 32 | 1824 | 0.056 | 0.1352 | 0.1418 |
| 4 | 32 | 3043 | 0.093 | 0.1881 | 0.2032 |
| 4 | 32 | 5076 | 0.155 | 0.2662 | 0.2602 |
| 4 | 32 | 8469 | 0.258 | 0.3130 | 0.3094 |
| 4 | 32 | 14128 | 0.431 | 0.3567 | 0.3500 |
| 4 | 32 | 23570 | 0.719 | 0.3973 | 0.3828 |
| 4 | 32 | 39322 | 1.200 | 0.3964 | 0.4087 |

Figure: `figures/bit_error_vs_load.png`

## E2. Error-free capacity scaling (50% of stored patterns fully stable)

P* comes from a log-space bisection on a Monte Carlo stable fraction. The theory value assumes independent bit errors: (1 - p_err)^N = 1/2.

| F | N | P* sim | P* theory | sim/theory |
|---|---|---|---|---|
| x^2 | 32 | 8.5 | 8.6 | 0.99 |
| x^2 | 64 | 13.5 | 12.9 | 1.04 |
| x^2 | 128 | 21.5 | 20.5 | 1.05 |
| x^2 | 256 | 34.5 | 34.0 | 1.02 |
| x^2 | 512 | 60.5 | 57.8 | 1.05 |
| x^3 | 16 | 32.5 | 27.1 | 1.20 |
| x^3 | 24 | 59.5 | 50.7 | 1.17 |
| x^3 | 32 | 88.5 | 80.3 | 1.10 |
| x^3 | 48 | 158.5 | 156.3 | 1.01 |
| x^3 | 64 | 260.0 | 253.3 | 1.03 |
| x^4 | 12 | 38.5 | 42.5 | 0.91 |
| x^4 | 16 | 79.5 | 86.2 | 0.92 |
| x^4 | 24 | 234.0 | 242.0 | 0.97 |
| x^4 | 32 | 468.5 | 512.2 | 0.91 |
| x^4 | 48 | 1516.0 | 1498.6 | 1.01 |

Log-log slope of P* vs N. The asymptotic exponent is n-1; the ln N factor pulls finite-size slopes below it:

| n | asymptotic exponent | sim slope | theory slope | prefactor ~ |
|---|---|---|---|---|
| 2 | 1 | 0.70 | 0.69 | 1/(2*1*ln N) |
| 3 | 2 | 1.48 | 1.61 | 1/(2*3*ln N) |
| 4 | 3 | 2.64 | 2.57 | 1/(2*15*ln N) |

Exponential F(x)=e^x:

| N | P* sim | P* Gaussian theory | P*/2^N |
|---|---|---|---|
| 8 | 46.5 | 59.8 | 0.1816 |
| 10 | 139.5 | 195 | 0.1362 |
| 12 | 411.5 | 666 | 0.1005 |
| 14 | 1484.0 | 2.34e+03 | 0.0906 |
| 16 | 4991.5 | 8.4e+03 | 0.0762 |
| 18 | 18208.4 | 3.05e+04 | 0.0695 |
| 20 | 64305.8 | 1.12e+05 | 0.0613 |

Fitted growth: sim P* ~ 2^(0.874 N); Gaussian SNR theory predicts 2^(0.908 N) (asymptotically log2(e^2/cosh 2) = 0.974).
Figure: `figures/capacity_scaling.png`

## E3. Pairwise learning rules under correlated (biased) patterns

Bias a means every bit is +1 with probability (1+a)/2, so any two patterns have an expected overlap a^2. P* is again the 50%-fully-stable load.

| rule (N=128), P*/N | a=0.0 | a=0.2 | a=0.4 | a=0.6 |
|---|---|---|---|---|
| Hebbian | 0.168 | 0.137 | 0.051 | 0.027 |
| Storkey | 0.512 | 0.488 | 0.316 | 0.105 |
| Pseudo-inverse (zero diag) | all P < N | all P < N | all P < N | all P < N |

Reference points: the Hebbian theory gives P*/N = 0.161 at a=0, and Storkey's asymptotic N/sqrt(2 ln N) gives 0.321. The pseudo-inverse stores every P < N pattern at any bias: with the diagonal kept, W xi = xi; with it zeroed, the field is (1 - w_ii) xi and w_ii ~ P/N < 1. Its advantage is invariance to correlation, and it costs an O(P^3) solve plus non-local learning.

## E4. Hebbian retrieval transition vs the AGS replica prediction

Start at a stored pattern, run asynchronous dynamics to a fixed point, and record the final overlap m. The replica-symmetric T=0 theory says m jumps from about 0.97 to about 0 at alpha_c.

| alpha | AGS m | sim m (N=500) | sim m (N=2000) |
|---|---|---|---|
| 0.06 | 1.000 | 1.000 | 1.000 |
| 0.07 | 1.000 | 0.999 | 1.000 |
| 0.08 | 1.000 | 1.000 | 0.999 |
| 0.09 | 0.999 | 1.000 | 0.999 |
| 0.10 | 0.998 | 0.996 | 0.999 |
| 0.11 | 0.996 | 0.997 | 0.995 |
| 0.12 | 0.993 | 0.993 | 0.990 |
| 0.13 | 0.987 | 0.995 | 0.987 |
| 0.14 | 0.000 | 0.836 | 0.889 |
| 0.15 | 0.000 | 0.969 | 0.868 |
| 0.16 | 0.000 | 0.827 | 0.756 |
| 0.17 | 0.000 | 0.805 | 0.644 |
| 0.18 | 0.000 | 0.482 | 0.320 |
| 0.19 | 0.000 | 0.535 | 0.320 |
| 0.20 | 0.000 | 0.575 | 0.243 |

alpha_c (computed) = 0.13791; m(alpha_c^-) = 0.9677. Above alpha_c the theory has no retrieval branch (shown as 0). The state the dynamics actually reaches there keeps a partial overlap, and finite N pushes the apparent transition to higher alpha.
Figure: `figures/ags_transition.png`

## E5. Basins of attraction at a fixed load

N=256, P=25 (load 0.1). Corrupt a stored pattern by a fraction f of bits, run asynchronous dynamics, and count a success when the final overlap is >= 0.98. Each trial uses a fresh pattern set.

| model | f=0.0 | f=0.05 | f=0.1 | f=0.15 | f=0.2 | f=0.25 | f=0.3 | f=0.35 | f=0.4 | f=0.45 |
|---|---|---|---|---|---|---|---|---|---|---|
| Hebbian | 1.00 | 0.95 | 1.00 | 0.97 | 0.97 | 0.85 | 0.72 | 0.45 | 0.07 | 0.00 |
| Pseudo-inverse (diag kept) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.20 | 0.03 |
| Pseudo-inverse (zero diag) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.85 | 0.45 | 0.00 |
| Dense x^3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 |

Figure: `figures/basins.png`

## E6. Continuous modern Hopfield: exponential capacity and the role of beta

d=64 with +/-1 patterns (norm M=8). The query is a stored pattern with 13 bits flipped (20%), and we apply a single update f(xi) = X^T softmax(beta X xi). Reported: nearest-neighbour success (the beta→∞ limit, vs exact theory), and the mean relative error ||f(xi)-x_i||/||x_i|| at each beta.

| P | NN success sim | NN success theory | rel err β=0.02 | rel err β=0.05 | rel err β=0.2 | rel err β=1.0 |
|---|---|---|---|---|---|---|
| 16 | 1.000 | 1.000 | 0.895 | 0.713 | 0.025 | 0.000 |
| 32 | 1.000 | 1.000 | 0.939 | 0.823 | 0.051 | 0.000 |
| 64 | 1.000 | 1.000 | 0.965 | 0.894 | 0.092 | 0.000 |
| 128 | 1.000 | 1.000 | 0.975 | 0.930 | 0.173 | 0.000 |
| 256 | 1.000 | 1.000 | 0.981 | 0.950 | 0.287 | 0.000 |
| 512 | 1.000 | 1.000 | 0.985 | 0.961 | 0.427 | 0.000 |
| 1024 | 1.000 | 0.999 | 0.987 | 0.966 | 0.580 | 0.001 |
| 2048 | 1.000 | 0.998 | 0.987 | 0.968 | 0.706 | 0.002 |
| 4096 | 0.990 | 0.996 | 0.988 | 0.970 | 0.790 | 0.009 |
| 8192 | 1.000 | 0.992 | 0.988 | 0.971 | 0.840 | 0.003 |
| 16384 | 0.980 | 0.985 | 0.988 | 0.971 | 0.868 | 0.023 |
| 32768 | 0.975 | 0.970 | 0.988 | 0.971 | 0.882 | 0.039 |
| 65536 | 0.935 | 0.940 | 0.988 | 0.971 | 0.889 | 0.078 |
| 131072 | 0.875 | 0.884 | 0.988 | 0.971 | 0.893 | 0.148 |

Retrieval bound ||f(xi)-x_i|| <= 2M eps. We check it at P=256 and beta=1 over 200 probes per corruption level. 'Informative' means the bound is below M, i.e. it certifies retrieval to within one pattern norm:

| flipped bits | violations | fraction informative |
|---|---|---|
| 0 | 0 | 1.00 |
| 1 | 0 | 0.77 |
| 2 | 0 | 0.00 |
| 4 | 0 | 0.00 |
| 13 | 0 | 0.00 |

Figure: `figures/modern_retrieval.png`

Exact theory for the load at which nearest-pattern retrieval succeeds 50% of the time, at the same corruption fraction:

| d | flips | P50 | log2(P50)/d |
|---|---|---|---|
| 16 | 3 | 65.8 | 0.378 |
| 24 | 5 | 210 | 0.322 |
| 32 | 6 | 2.59e+03 | 0.354 |
| 48 | 10 | 2.25e+04 | 0.301 |
| 64 | 13 | 7.37e+05 | 0.305 |
| 96 | 19 | 7.44e+08 | 0.307 |
| 128 | 26 | 1.78e+11 | 0.292 |


## E7. What the capacity costs: parameters and compute at N = 64

The same energy can be stored two ways. The tensor form keeps the n-body synaptic tensor (C(N, n) distinct entries). The pattern form keeps the P x N matrix of +/-1 patterns (hidden units). A full asynchronous sweep costs N * C(N-1, n-1) multiply-adds in tensor form and about N * P in pattern form. P* is the simulated value when N is on the simulated grid, otherwise the theory value.

| F | P* | source | bits stored (N P*) | tensor params | bits / tensor param | sweep MACs (tensor) | sweep MACs (pattern) |
|---|---|---|---|---|---|---|---|
| x^2 | 13 | sim | 863 | 2.02e+03 | 0.428 | 4.03e+03 | 863 |
| x^3 | 260 | sim | 1.66e+04 | 4.17e+04 | 0.399 | 1.25e+05 | 1.66e+04 |
| x^4 | 3240 | theory | 2.07e+05 | 6.35e+05 | 0.326 | 2.54e+06 | 2.07e+05 |

Superlinear capacity is paid for in parameters. In pattern form the memory is the patterns, stored at exactly 1 bit per parameter, and update cost grows linearly in P. In tensor form the parameter count grows as N^n / n!, so bits per parameter stays well below 1 and shrinks roughly like 1/ln N. The exponential model grows as 2^(0.87 N) in the simulated range and has no finite tensor form. Its pattern form is a softmax over all stored patterns, the same cost as the continuous model in E6.

_Total wall time: 119.8 s._
