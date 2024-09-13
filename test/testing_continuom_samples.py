import sys
sys.path.append('.')

import numpy as np

from searchs.bayesian import continuom_samples

def generate_discrete_values(start, end, num_points, dtype=None):
    return np.linspace(start, end, num=num_points, dtype=dtype)
    

discrete_values = {
    'param1': generate_discrete_values(0.1, 1000, 20),
    'param2': generate_discrete_values(0.01, 100, 20, dtype=int),
}


print('starting')
for k, v in discrete_values.items():
    discrete_values[k] = continuom_samples(v)
print('finished')

print(discrete_values)