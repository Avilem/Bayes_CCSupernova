import sys
import os

Path = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(Path, "src")
sys.path.insert(0, SRC)

# Import other python libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import bilby
import pandas as pd
import gwpy
from gwpy.timeseries import TimeSeries
import os
from scipy import signal
import h5py

# Import PyCBC functions
from pycbc import types
import pycbc.noise
import pycbc.psd.estimate as psd_est
#import pycbc.psd
from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.filter import sigma
from pycbc.filter import match

import preprocess as pp
import sn_library as sn
import CBS as TCB
import initsampler as ins
import merge_results as mr

fs = 16384
Dinkpc = 1

richers_catalog = pp.load_richers(Path +'/data/Richers/',True,fs,Dinkpc,True)

tini = -5
tend = 5

results_directory =f'PE_RIC_O3_{Dinkpc}_filtered'
for s in range(107,126):
  signal_name = f'RIC_nsls_signal_{s}'
  print(f"Processing {signal_name}...")

  hs = pp.prepare_signal(richers_catalog,tini,tend,s,fs, False)
  x = pp.generate_x(hs,tend-tini,tini,tend,fs)
  w = pp.whitening(x)
  likelihood, priors = ins.init_sampler(w,'Uniform',TCB.CoreBounceSignal,Dinkpc)

  result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler='bilby_mcmc',
            nsamples=1000,
            label=signal_name,
            outdir=results_directory,
            verbose=True,
        )
  mr.create_csv(result,Dinkpc,signal_name,results_directory)

print("Parameter estimation for all signals completed.")

