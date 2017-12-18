import pandas as pd
import numpy as np

import os


def preprocess_v1(filename, preprocess = True):
  required_columns = [ 'age', 'sex', 'driverdrowsy', 'alcinvol', 'druginv', 'dridistract', 'numoccs', 'prevacc', 'numfatal', 'year' ]

  files = os.listdir('all_data')
  files.sort(reverse=True)

  total_df = pd.DataFrame([], columns = required_columns)

  for file in files:
    year = int(file[-8:-4])

    if year <= 2009:
      break

    print "preprocessing ",file

    df = pd.read_csv('all_data/'+file,sep='\t')
    # Choose only drivers
    df = df[df['ptype'] == 1]
    # Choose if occupants is less than 6 - CAR
    df = df[df['numoccs']<= 6] 
    # Filtering only male and female
    df = df[df["sex"].isin([1,2])]
    if(preprocess):
        # Bin age Groups
        df.age[df["age"]<18] = 0
        df.age[(df["age"]>=18) & (df["age"]<25)] = 1
        df.age[(df["age"]>=25) & (df["age"]<65)] = 2
        df.age[df["age"]>=65] = 3
        # Bin Occupants
        df.numoccs[df['numoccs']>2] = 2
    
        # Alcohol involvement prediction
        if 'alcres' in df.columns:
          df['alcres'] = df['alcres'] > 0.1
          df['alcinvol'] = df['alcinvol'] == 1

          df['alcinvol'] = (df['alcres'] | df['alcinvol'])
        else:
          df['alcinvol'] = df['alcinvol'] == 1

        # Drug involvement prediction
        df['druginv'] = df['druginv'] == 1

        df['driverdrowsy'] = df['driverdrowsy'] == 1

        df['dridistract'] = df['dridistract'] > 0

    df['year'] = pd.Series(np.ones(df.shape[0])*int(file[-8:-4]), index=df.index)
    df['year'] = df['year'].astype('int64')

    df = df[required_columns]

    total_df = total_df.append(df)

  total_df.to_csv(filename, index=False)

preprocess_v1('preprocessed_data_2.csv')
preprocess_v1('preprocessed_data.csv', False)
