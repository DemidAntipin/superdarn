import os
import pandas as pd

def check_bz2_files(directory):
  for file in os.listdir(directory):
    if file.endswith('.bz2'):
      file_path=os.path.join(directory, file)
      try:
        df=pd.read_csv(file_path, compression='bz2')
      except pd.errors.EmptyDataError:
        print(file)
        print("error")
      except Exception as e:
        print("else")

directory_path='./'
check_bz2_files(directory_path)
