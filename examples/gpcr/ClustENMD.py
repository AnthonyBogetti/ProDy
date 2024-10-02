import numpy as np
import matplotlib.pyplot as plt
from prody import *

mol = parsePDB("2RH1")
clustenm = ClustENM()
clustenm.setAtoms(mol)
clustenm.run(cutoff=15, n_modes=5, rmsd=2, n_gens=5, n_confs=50, maxclust=10, sim=False, crk=10)
saveEnsemble(clustenm)
clustenm.writePDB()
