import pandas as pd
import sn_library as sn
import numpy as np
import matplotlib.pyplot as plt
import gwpy
import random
from gwpy.timeseries import TimeSeries
import h5py

from scipy import signal

# Import PyCBC functions
from pycbc import types
import pycbc.noise
import pycbc.psd.estimate as psd_est
#import pycbc.psd
from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.filter import sigma
from pycbc.filter import match


# This toolbox includes some functions for pre-processing data from Richers and Abylkairov catalogs

def load_abylkairov(path,plot,wave_type,fs,Dinkpc,filter):
  """
  path: String for the path where data from catalogs is stored
  plot: Boolean
  wave_type: 0 for GR signals 1 for GREP
  fs: Sampling frequency
  Dinkpc: Distance in kiloparsecs
  filter: Boolean. Use a Butterworth filter or not
  """
  df = pd.read_csv(path)
  if wave_type == 0:
    GR_waves = df[df['GR_or_GREP'] == 0]

  elif wave_type == 1:
    GR_waves = df[df['GR_or_GREP'] == 1]

  else:
    print('Choose either 0 (GR waves) or 1 (GREP waves)')

  #Convert Distance to meters
  kpc2m    = 3.08567758128e+19
  D = Dinkpc * kpc2m
  #Butterworth filter
  fcri = 800
  sos = signal.butter(N=2,Wn=fcri,btype='lowpass',fs=fs,output='sos')

  ab_catalog = {}
  for s in range(452):
    sig = GR_waves[GR_waves['sample_id'] == s]
    h_ab = np.array(sig['amplitude'])/(100*D) #Convert to meters
    t_ab = np.array(sig['t(ms)'])/1000  #Convert to seconds

    t_ab, h_ab     = sn.sn_resample_wave(t_ab, h_ab, fs) #Resample to fs sampling rate
    if filter:

      h = signal.sosfiltfilt(sos,h_ab)

      ab_catalog['h'+str(s)] = h
      ab_catalog['t'+str(s)] = t_ab
    else:
      ab_catalog['h'+str(s)] = h_ab
      ab_catalog['t'+str(s)] = t_ab


  if plot:
    for s in range(452):
      t,h = ab_catalog['t' + str(s)],ab_catalog['h' + str(s)]
      plt.plot(t,h)
      plt.xlim(-0.005,0.007)
      plt.xlabel('Time (s)',fontsize=14)
      plt.ylabel('Strain',fontsize=14)
      plt.title(f'Abylkairov Signals @ {Dinkpc} kpc ')

  return ab_catalog

def load_richers(path,plot,fs,Dinkpc,filter):
  """
  path: Path where the data is stored
  plot: Boolean
  fs: sampling frequency
  Dinkpc: Distance in kiloparsecs
  filter: Boolean. Apply a Butterworth filter or not
  """
  #Convert Distance to meters
  kpc2m    = 3.08567758128e+19
  D = Dinkpc * kpc2m

  #Butterworth filter
  fcri = 800
  sos = signal.butter(N=2,Wn=fcri,btype='lowpass',fs=fs,output='sos')

  # Get 126 signals from name AXwY.00_Z. X: Differential rotation, Y: Angular velocity, Z: EOS.

  EOS    = ['SFHo', 'SFHx', 'LS220', 'BHBLP', 'HSDD2', 'GShenFSU2.1']
  name   = [10000, 1268, 300, 467, 634]

  Signal_Name = []

  for i in range (5):
      if   (name[i] == 10000):  Omega_init, Omega_end = 1 , 3
      elif (name[i] == 1268):   Omega_init, Omega_end = 1 , 5
      elif (name[i] == 300):    Omega_init, Omega_end = 3 , 11
      elif (name[i] == 467):    Omega_init, Omega_end = 3 , 6
      elif (name[i] == 634):    Omega_init, Omega_end = 2 , 6

      for j in range (Omega_init, Omega_end):
          for k in range(len(EOS)):
              fullname = 'A'+ str(name[i])+ 'w' + str(j+1) + '.00_' + EOS[k]
              Signal_Name.append(fullname)


  richers_catalog = {}
  for sign in range(126):
    isignal = sign # Buenas: 17,51,121, # ********** SET THIS **********

    namereal      = Signal_Name[isignal]

      # -------------------------------
    # Get signal from the catalog
    FileInfo         = h5py.File(path + "GWdatabase.h5", 'r')
    group            = FileInfo['waveforms']
    subgroup         = group[namereal]
    h_cat            = np.array( subgroup[list(subgroup.keys())[3]] )
    t_cat            = np.array( subgroup[list(subgroup.keys())[4]] )

    t_cat, h_cat     = sn.sn_resample_wave(t_cat, h_cat, fs)

    if filter:

      h = signal.sosfiltfilt(sos,h_cat)

      richers_catalog['h'+str(sign)] = h/(D*100)
      richers_catalog['t'+str(sign)] = t_cat
    else:
      richers_catalog['h'+str(sign)] = h_cat/(D*100)
      richers_catalog['t'+str(sign)] = t_cat


  if plot:
    for s in range(126):
      t,h = richers_catalog['t' + str(s)],richers_catalog['h' + str(s)]
      plt.plot(t,h)
      plt.xlim(-0.005,0.007)
      plt.xlabel('Time (s)',fontsize=14)
      plt.ylabel('Strain',fontsize=14)
      plt.title(f'Richers Signals @ {Dinkpc} kpc ')

  return richers_catalog

def prepare_signal(catalog,tini,tend,s,fs):
  """
  This function prepares the signal for injecting noise
  catalog: Dictionary with the time and strain data
  tini: Initial time of x
  tend: End time of x
  s: signal number
  fs: sampling frequency
  """
  t, h = catalog['t' + str(s)], catalog['h' + str(s)]
# Generate strain time-series
  hs = pycbc.types.TimeSeries(initial_array=h, delta_t=1.0/fs, epoch=t[0])

  # Generate a zeros time series to inject signal
  tzeros = np.arange(tini, tend, 1/fs)
  hzeros = pycbc.types.TimeSeries(initial_array=tzeros*0, delta_t=1.0 / fs, epoch=tzeros[0])

  # Inject noise in signal
  hs = hzeros.inject(hs, copy=True)
  return hs

def generate_x(hs,tini,tend,fs):
  """
  Generate x = signal + noise, taking real noise from O3b

  hs: Time-series object from pycbc of the signal
  tini: initial time
  tend: end time
  fs: sampling frequency
  """
  gps = 1256677376 + random.randint(0, 4000)
  segment = (int(gps), int(gps) + tend-tini)

  ldata = TimeSeries.fetch_open_data('L1', *segment, sample_rate=fs, verbose=False)
  nt = pycbc.types.TimeSeries(initial_array=ldata.value, delta_t=1.0 / fs)
  nt.start_time = tini

  x = hs + nt
  return x
def whitening(x):
  """
  This function whitens the observation x = signal + noise and smoothens it with a band-pass filter
  x: Time series object from pycbc
  """
  welch = psd_est.welch
  if callable(psd_est.interpolate):
      interpolate = psd_est.interpolate
  else:
  # Handle the version of the module in python, because it might vary
      interpolate = psd_est.interpolate.interpolate

  psd = interpolate(welch(x), 1.0 / x.duration)
  white_strain = ((x.to_frequencyseries() / psd ** 0.5).to_timeseries())*10e-24
  white_strain_cropped = white_strain.crop(1, 1)
  smooth = highpass_fir(white_strain_cropped, 35, 8)
  smooth = lowpass_fir(smooth, 300, 7)

  return smooth
