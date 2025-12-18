import sys
import os

Path = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(Path, "src") #Add the src folder to the path
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

# Import PyCBC functions
from pycbc import types
import pycbc.noise
import pycbc.psd.estimate as psd_est
#import pycbc.psd
from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.filter import sigma
from pycbc.filter import match

# Import other tools
import preprocess as pp
import sn_library as sn
import CBS as TCB
import initsampler as ins
import merge_results as mr

fs = 16384
Dinkpc = 1

ab_catalog = pp.load_abylkairov(Path + '/data/Abylkairov/Abylkairov_catalog.csv',False,0,fs,Dinkpc,True)

tini = -5
tend = 5
s = 1

results_directory =f'PE_AB_O3_{Dinkpc}_filtered'
for s in range(452):
  signal_name = f'{Dinkpc}_signal_{s}'
  print(f"Processing {signal_name}...")

  hs = pp.prepare_signal(ab_catalog,tini,tend,s,fs, False)
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

