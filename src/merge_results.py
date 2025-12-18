import pandas as pd
import bilby
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_csv(results,signal_name,results_directory):
  """
  This function creates and saves csv files into the given directory
  results: Result object that outputs from run_sampler in bilby.
  signal_name: String usually includes the number of the signal, the catalog and the distance
  results_directory: String for the directory in which csv files will be saved
  """
  values = {
          'beta_est': results.posterior['beta'].mean(),
          'beta_un': results.posterior['beta'].std(),
          'alpha_est': results.posterior['alpha'].mean(),
          'alpha_un': results.posterior['alpha'].std(),
          'tau_est': results.posterior['tau'].mean(),
          'tau_un': results.posterior['tau'].std(),
          's_est': results.posterior['s'].mean(),
          's_un': results.posterior['s'].std(),
      }
  df = pd.DataFrame.from_dict(values, orient='index')
  final = df.T
  output_path = os.path.join(results_directory, f"{signal_name}.csv")
  final.to_csv(output_path)

def merge_csv(n,Dinkpc,signal_name,path,results_directory):
  """
  Function for merging all the csv files into one in order.

  n: Total number of signals
  Dinkpc: Distance in kiloparsecs
  signal_name: string 
  path: string
  results_directory: string
  """
  rows = []

  # Reads the first file
  first_df = pd.read_csv(path + signal_name + ".csv")
  num_cols = first_df.select_dtypes(include='number').columns.tolist()

  # Add a column for the file number
  fin_cols = ["file"] + num_cols

  for k in range(n):
      # Read each file
      df = pd.read_csv(f"{Dinkpc}signal_{k}.csv")

    # Get num values
      vals = df.select_dtypes(include='number').values.flatten()

    #Create a row with the file's name and data
      row = [f"{Dinkpc}signal_{k}"] + list(vals)

      rows.append(row)

  # Create DF with the first file in the header
  final_df = pd.DataFrame(rows, columns=fin_cols)

  output_path = os.path.join(results_directory, results_directory + "_merged.csv")
  final_df.to_csv(output_path, index=False)

def read_n_plot_results(file,figs_dir,Dinkpc):
  resultados = pd.read_csv(file)
  abyl = pd.read_csv('/Abylkairov/Abylkairov_catalog.csv')  #Load catalog
  ab_0 = abyl[(abyl['t(ms)'] == 0) & (abyl['GR_or_GREP'] == 0)]  #Take GR waveforms and the values of the parameters at bounce t = 0

  # Change the names of the EOS and add a column
  eos_map = {
      0: 'SFHo',
      1: 'LS220',
      2: 'HSDD2',
      3: 'GShenFSU2.1'
  }

  ab_0 = ab_0.copy()

  ab_0['eos_name'] = ab_0['EOS'].map(eos_map)

  resultados_eos = resultados.copy()
  resultados_eos['eos_name'] = ab_0['eos_name'].reset_index(drop=True)
  resultados_eos['T/|W|'] = ab_0['T/|W|'].reset_index(drop=True)
  resultados_eos['fpeak'] = ab_0['f_peak'].reset_index(drop=True)
  resultados_eos['Deltah'] = ab_0['D Delta h'].reset_index(drop=True)
 

  #Estimation quality
  sns.scatterplot(data = resultados_eos,x='T/|W|',y='beta_est', hue='eos_name')
  plt.plot([0.02,0.18],[0.02,0.18],color='tab:gray',linestyle='--',label=r'$\hat{\beta} = \beta$')
  plt.grid(True)
  plt.xlabel(r'$\beta$',fontsize=14)
  plt.ylabel(r'$\hat{\beta}$',fontsize=14)
  plt.title(f'$\beta$ estimation @ {Dinkpc}', fontsize = 16)
  plt.legend(title='Equation of State')
  plt.savefig(figs_dir + f'beta_est_{Dinkpc}.png')

  sns.scatterplot(data = resultados_eos,x='beta_est',y='alpha_est', hue='eos_name')
  plt.grid(True)
  plt.title(fr'$\beta$ vs $\alpha$ @ {Dinkpc}', fontsize=16)
  plt.xlabel(r'$\beta$',fontsize=14)
  plt.ylabel(r'$\alpha$',fontsize=14)
  plt.legend(title='Equation of State')
  plt.savefig(figs_dir + fr'beta_vs_alpha_{Dinkpc}.png')

  return resultados_eos

