import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


number_of_samples = 3
size = 60
triplets_x = np.random.random((number_of_samples, 4, size))
vector_names = ['Index/Feature Identifier', 'Time', 'Value', 'Mask']
colors = ['blue', 'green', 'red', 'purple']

fig = plt.figure(figsize=(12, 8))
axes = [fig.add_subplot(2, 2, i + 1, projection='3d') for i in range(4)]

for i in range(4):
    ax = axes[i]
    for sample_idx in range(number_of_samples):
        x = np.arange(size)  # seq pos
        y = np.full(size, sample_idx)
        z = triplets_x[sample_idx, i, :]  # values

        ax.plot(x, y, z, color=colors[i], label=f'Sample {sample_idx + 1}')

    ax.set_title(f'{vector_names[i]} Vector')
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Sample Index')
    ax.set_zlabel('Value')

plt.tight_layout()
plt.show()
