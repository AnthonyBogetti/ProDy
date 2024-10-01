import numpy as np
import matplotlib.pyplot as plt
from prody import *

mol = parsePDB("1AKE")
clustenm = ClustENM()
clustenm.setAtoms(mol)
clustenm.run(cutoff=15, n_modes=5, rmsd=1, n_gens=5, n_confs=50, maxclust=10, sim=False)
saveEnsemble(clustenm)
clustenm.writePDB()
