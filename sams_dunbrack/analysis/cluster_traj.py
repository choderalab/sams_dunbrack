import mdtraj as md
import numpy as np
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

pdbid = '5UG9'
experiment = 'both'
iteration = 250000
traj = md.load(f'{experiment}_{iteration}_traj.dcd', top=f'{pdbid}_minimized.pdb')

distances = np.empty((traj.n_frames, traj.n_frames))
for i in range(traj.n_frames):
    print(i)
    distances[i] = md.rmsd(traj, traj, i)
print('Max pairwise rmsd: %f nm' % np.max(distances))

# Clustering only accepts reduced form. Squareform's checks are too stringent
#assert np.all(distances - distances.T < 1e-6)
reduced_distances = squareform(distances, checks=False)

linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method='average')

plt.title('RMSD Average linkage hierarchical clustering')
_ = scipy.cluster.hierarchy.dendrogram(linkage, no_labels=True, count_sort='descendent')
plt.show()
