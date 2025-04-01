import os
import bz2
from glob import glob

def find_files(directory):
  return glob(os.path.join(directory, '*.bz2'))

directory='data/'

def distribute_lines(directory):
    files = find_files(directory)

    for file in files:
        lines=pd.read_csv(file, compression='bz2', header=None, sep=r"\s+")
        lines = pd.read_csv(file, compression='bz2', header=None, sep=r"\s+")
        lines = lines.sample(frac=1, random_state=42).reset_index(drop=True)
        line_count = len(lines)
        
        train_count = int(line_count * 0.8)
        validation_count = int(line_count * 0.1)
        test_count = line_count - train_count - validation_count

        train_data = lines.iloc[:train_count]
        validation_data = lines.iloc[train_count:train_count + validation_count]
        test_data = lines.iloc[train_count + validation_count:]

        base_filename = os.path.splitext(os.path.basename(file))[0]

        train_filename = os.path.join(directory, 'train', base_filename")
        validation_filename = os.path.join(directory, 'validation', base_filename)
        test_filename = os.path.join(directory, 'test', base_filename)

        os.makedirs(os.path.dirname(train_filename), exist_ok=True)
        os.makedirs(os.path.dirname(validation_filename), exist_ok=True)
        os.makedirs(os.path.dirname(test_filename), exist_ok=True)

        if not train_data.empty:
            train_data.to_csv(train_filename, index=False, header=False)
        
        if not validation_data.empty:
            validation_data.to_csv(validation_filename, index=False, header=False)
        
        if not test_data.empty:
            test_data.to_csv(test_filename, index=False, header=False)

distribute_lines(directory)
