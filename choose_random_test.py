import os
import numpy as np
import csv


data_dir  = '/media/STORAGE/DATASETS/open-images/challenge2018/'
csvfile = './csv_files/test.csv'
images = os.listdir(data_dir)
chosen = np.random.choice(images, 64, replace=False)
print(chosen)

with open(csvfile, 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(chosen)

# with open(csvfile, 'r') as f:
#     reader = csv.reader(f)
#     for i, row in enumerate(reader):
#         print('Row', i, row)
