# ALE-from-slits
Code to simulate the aggregate Loewner evolution (ALE) process started from a non-trivial initial configuration of slits. You can specify your own configuration or allow the script to choose a random one.

Currently the parameters (number of slits, their locations and lengths, $\eta$ for the ALE, etc.) have to be typed into `ale-script.py`, or you can allow it to choose the parameters randomly (the distribution can be changed by modifying `get_random_configuration()` ). This is set around line `211`.
I'll make it more user-friendly when I get a chance.

There's also a Jupyter Notebook which does more-or-less the same thing, but with extensive explanations of the method (based on Pearce, 1991 [https://epubs.siam.org/doi/10.1137/0912013]).

--

Note that the $\sigma$ is a bit larger than it should be for the convergence theorems in the paper [https://arxiv.org/abs/2304.04417] to hold (or at least for the proofs to be valid), but numerically it's not very reliable if we take a really tiny $\sigma$.
