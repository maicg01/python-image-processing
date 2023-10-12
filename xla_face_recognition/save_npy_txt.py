import numpy as np

# Tải file npy
data = np.load('/home/maicg/Desktop/database_local/cerberus_ada_r50.npy')

# Lưu dữ liệu thành file txt
output_file = 'output2.txt'
np.savetxt(output_file, data, delimiter='\n', fmt='%s')