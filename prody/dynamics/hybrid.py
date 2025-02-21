from itertools import product
from multiprocessing import cpu_count, Pool
from collections import OrderedDict
from os import chdir, mkdir, remove
from os.path import isdir
from sys import stdout

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage

from prody import LOGGER
from .anm import ANM
from .gnm import GNM, ZERO
from .rtb import RTB
from .imanm import imANM
from .exanm import exANM
from .editing import extendModel
from .sampling import traverseMode, sampleModes
from prody.atomic import AtomGroup
from prody.measure import calcTransformation, applyTransformation, calcRMSD
from prody.ensemble import Ensemble
from prody.proteins import writePDB, parsePDB, writePDBStream, parsePDBStream
from prody.utilities import createStringIO, importLA, mad

la = importLA()
norm = la.norm

__all__ = ['HYBRID']

class HYBRID():

    def __init__(self):

        self._atoms = None
        self._idx_ca = None
        self._n_ca = None
        self._n_atoms = None
        self._indices = None
        self._topology = None
        self._positions = None
        self._potential = None
        self._maxclust = 5
        self._forcefield = ('amber14-all.xml', 'implicit/gbn2.xml')
        self._anm = ANM()
        self._ca = None

        try:
            import pdbfixer
            import openmm
        except ImportError:
            raise ImportError('Please install PDBFixer.')

    def prep_sim(self, atoms):

        from pdbfixer import PDBFixer
        from openmm.app import PDBFile

        LOGGER.info('Fixing the structure...')
        atoms = atoms.select('not hetatm')
        stream = createStringIO()
        writePDBStream(stream, atoms)
        stream.seek(0)
        fixed = PDBFixer(pdbfile=stream)
        stream.close()

        fixed.missingResidues = {}
        fixed.findNonstandardResidues()
        fixed.replaceNonstandardResidues()
        fixed.removeHeterogens(False)
        fixed.findMissingAtoms()
        fixed.addMissingAtoms()
        fixed.addMissingHydrogens(7.)

        stream = createStringIO()
        PDBFile.writeFile(fixed.topology, fixed.positions,
                          stream, keepIds=True)
        stream.seek(0)
        self._atoms = parsePDBStream(stream)
        stream.close()

        self._topology = fixed.topology
        self._positions = fixed.positions

        self._idx_ca = self._atoms.ca.getIndices()
        self._n_ca = self._atoms.ca.numAtoms()
        self._n_atoms = self._atoms.numAtoms()
        self._indices = None
        LOGGER.info('Structure fixed successfully.')

    def minimize(self, n_cycles=1000, log=True, save=False):

        from openmm import Platform, LangevinIntegrator, Vec3
        from openmm.app import Modeller, ForceField, \
            CutoffNonPeriodic, PME, Simulation, HBonds, NoCutoff, PDBFile
        from openmm.unit import angstrom, nanometer, picosecond, \
            kelvin, Quantity, molar

        if log:
            LOGGER.info('Performing energy minimization...')
        forcefield = ForceField(*self._forcefield)

        system = forcefield.createSystem(self._topology, 
                                         nonbondedMethod=NoCutoff,
                                         nonbondedCutoff=99.9*nanometer, 
                                         constraints=HBonds)
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 
                                        0.002*picosecond)
        simulation = Simulation(self._topology, system, integrator)
        simulation.context.setPositions(self._positions)
        simulation.minimizeEnergy(maxIterations=n_cycles)
        self._positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        self._potential = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        if log:
            LOGGER.info('Energy minimization successfully completed.')
        if save:
            PDBFile.writeFile(self._topology, self._positions, 
                              open("min.pdb", 'w'))

    def md(self, heat_steps=10000, prod_steps=50000000, log=True, save=False):

        from openmm import Platform, LangevinIntegrator, Vec3
        from openmm.app import Modeller, ForceField, \
            CutoffNonPeriodic, PME, Simulation, HBonds, NoCutoff, PDBFile
        from openmm.unit import angstrom, nanometer, picosecond, \
            kelvin, Quantity, molar

        if log:
            LOGGER.info('Performing MD...')
        forcefield = ForceField(*self._forcefield)

        system = forcefield.createSystem(self._topology, 
                                         nonbondedMethod=NoCutoff,
                                         nonbondedCutoff=99.9*nanometer, 
                                         constraints=HBonds)
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 
                                        0.002*picosecond)
        simulation = Simulation(self._topology, system, integrator)
        simulation.context.setPositions(self._positions)
        simulation.step(heat_steps)
        simulation.step(prod_steps)
        self._positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        if log:
            LOGGER.info('MD successfully completed.')
        if save:
            PDBFile.writeFile(self._topology, self._positions, 
                              open("md.pdb", 'w'))

    def build_anm(self, log=True):

        from openmm.unit import angstrom

        if log:
            LOGGER.info('Building ANM...')
        pos = self._positions.value_in_unit(angstrom)[:self._topology.getNumAtoms()]

        tmp = self._atoms.copy()
        tmp.setCoords(pos)
        self._ca = tmp[self._idx_ca]
                            
        self._anm.buildHessian(self._ca)
        if log:
            LOGGER.info('ANM successfully built.')

    def sample(self, conf, n_confs, n_modes, rmsd):

        tmp = self._atoms.copy()
        tmp.setCoords(conf)
        cg = tmp[self._idx_cg]

        anm_cg = self.build_anm(cg)

        anm_cg.calcModes(n_modes)

        anm_ex, atoms_all = extendModel(anm_cg, cg, tmp)
        ens_ex = sampleModes(anm_ex, atoms=tmp, n_confs=n_confs, rmsd=rmsd)
        coordsets = ens_ex.getCoordsets()

    def rmsds(self, coords):

        tmp = coords.reshape(-1, 3 * self._n_cg)

        return pdist(tmp) / np.sqrt(self._n_cg)

    def hc(self, arg):

        rmsds = self.rmsds(arg)
        link = linkage(rmsds, method='average')
        hcl = fcluster(link, t=self._maxclust,
                           criterion='maxclust') - 1
        return hcl

    def centroid(self, arg):

        if arg.shape[0] > 2:
            rmsds = self.rmsds(arg)
            sim = np.exp(- squareform(rmsds) / rmsds.std())
            idx = sim.sum(1).argmax()
            return idx
        else:
            return 0

    def centers(self, *args):

        nl = np.unique(args[1])
        idx = OrderedDict()
        for i in nl:
            idx[i] = np.where(args[1] == i)[0]

        wei = [idx[k].size for k in idx.keys()]
        centers = np.empty(nl.size, dtype=int)
        for i in nl:
            tmp = self.centroid(args[0][idx[i]])
            centers[i] = idx[i][tmp]

        return centers, wei

    def generate(self, confs, **kwargs):

        tmp = [self.sample(conf) for conf in confs]
        tmp = [r for r in tmp if r is not None]

        confs_ex = np.concatenate(tmp)
        confs_cg = confs_ex[:, self._idx_cg]

        label_cg = self._hc(confs_cg)
        centers, wei = self._centers(confs_cg, label_cg)
        confs_centers = confs_ex[centers]
        
        if len(confs_cg) > 1:
            label_cg = self._hc(confs_cg)
            centers, wei = self._centers(confs_cg, label_cg)
            confs_centers = confs_ex[centers]
        else:
            confs_centers, wei = confs_cg, [len(confs_cg)]

        return confs_centers, wei

    def run_anmd(self, skip_modes=0, n_modes=2, n_steps=5, rmsd=1, save=True):
        
        from openmm.app import PDBFile
        from openmm.unit import angstrom

        LOGGER.info('Starting classic ANMD...')

        self.build_anm(log=False)
        self._anm.calcModes(n_modes=n_modes)

        pos = self._positions.value_in_unit(angstrom)[:self._topology.getNumAtoms()]
        tmp = self._atoms.copy()
        tmp.setCoords(pos)

        anm_ex, atoms_all = extendModel(self._anm, self._ca, tmp)
        anm_ex._indices = self._anm.getIndices()
        eval_0 = self._anm[0].getEigval()

        for i in range(skip_modes, n_modes):

            modeNum = anm_ex.getIndices()[i]

            eval_i = self._anm[i].getEigval()
            sc_rmsd = ((1/eval_i)**0.5/(1/eval_0)**0.5)*rmsd
            traj_aa = traverseMode(anm_ex[i], atoms_all, n_steps=n_steps,
                                   rmsd=rmsd)
            traj_aa.setAtoms(atoms_all)

            num_confs = traj_aa.numConfs()

            target_ensemble = Ensemble('mode {0} ensemble'.format(modeNum))
            target_ensemble.setAtoms(atoms_all)
            target_ensemble.setCoords(atoms_all)
            
            LOGGER.info('Minimizing confs from mode %s.'%modeNum)
            for conf in traj_aa:
                self._positions = conf.getCoords()
                self.minimize(n_cycles=2, log=False)
                target_ensemble.addCoordset(self._positions)

            if save:
                writePDB("anmd_mode_%s"%str(modeNum), target_ensemble)

        LOGGER.info('ANMD completed successfully.')

    def run_clustenm(self, n_gens=2, rmsd=1, save=False):

        cycle = 0
        potentials = [self._potential]
        sizes = [1]
        new_shape = [1]
        for s in conformer.shape:
            new_shape.append(s)
        conf = conformer.reshape(new_shape)
        conformers = start_confs = conf
        keys = [(0, 0)]
                                                                                                  
        for i in range(1, n_gens+1):
            cycle += 1
            confs, weights = self.generate(start_confs)
            conf_list = []
            pot_list = []
            for conf in confs:
                self._positions = conf.getCoords()
                self.minimize()
                conf_list.append(self._positions)
                pot_list.append(self._potential)
                                                                                                  
            idx = np.logical_not(np.isnan(pot_list))
            weights = np.array(weights)[idx]
            pots = np.array(pot_list)[idx]
            confs = np.array(conf_list)[idx]
            idx = np.full(pot_list.size, True, dtype=bool)
                                                                                                  
            sizes.extend(weights[idx])
            potentials.extend(pot_list[idx])
            start_confs = self._superpose_cg(confs[idx])
                                                                                                  
            for j in range(start_confs.shape[0]):
                keys.append((i, j))
            conformers = np.vstack((conformers, start_confs))
                                                                                                  
        self._build(conformers, keys, potentials, sizes)

