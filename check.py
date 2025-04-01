import os
import pandas as pd

def empty_check_bz2_files(directory):
  for file in os.listdir(directory):
    if file.endswith('.bz2'):
      file_path=os.path.join(directory, file)
      try:
        df=pd.read_csv(file_path, compression='bz2')
      except pd.errors.EmptyDataError:
        os.remove(file_path)
      except Exception as e:
        print(e)

directory_path='data/'
empty_check_bz2_files(directory_path)
