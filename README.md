# ALE-from-slits
Code to simulate the aggregate Loewner evolution (ALE) process started from a non-trivial initial configuration of slits. You can specify your own configuration or allow the script to choose a random one.

Currently the parameters (number of slits, their locations and lengths, $\eta$ for the ALE, etc.) have to be typed into `ale-script.py`, or you can allow it to choose the parameters randomly (the distribution can be changed by modifying `get_random_configuration()` ).
I'll make this more user-friendly when I get a chance.

There's also a Jupyter Notebook which does more-or-less the same thing.
