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
        self._reference = None
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

    def minimize(self, n_cycles=1000, ref=True, hw='cpu', log=True, save=False):

        from openmm import Platform, LangevinIntegrator, Vec3
        from openmm.app import Modeller, ForceField, \
            CutoffNonPeriodic, PME, Simulation, HBonds, NoCutoff, PDBFile
        from openmm.unit import angstrom, nanometer, picosecond, \
            kelvin, Quantity, molar, kilojoule_per_mole

        if log:
            LOGGER.info('Performing energy minimization...')
        forcefield = ForceField(*self._forcefield)

        system = forcefield.createSystem(self._topology, 
                                         nonbondedMethod=NoCutoff,
                                         nonbondedCutoff=99.9*nanometer, 
                                         constraints=HBonds)
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 
                                        0.002*picosecond)

        if hw == 'cpu': 
            platform = Platform.getPlatformByName('CPU')
        elif hw == 'gpu':
            platform = Platform.getPlatformByName('CUDA')
        simulation = Simulation(self._topology, system, integrator, platform)
        simulation.context.setPositions(self._positions)
        simulation.minimizeEnergy(maxIterations=n_cycles)
        self._positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        self._potential = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        if ref:
            self._reference = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        if log:
            LOGGER.info('Energy minimization successfully completed.')
        if save:
            PDBFile.writeFile(self._topology, self._positions, 
                              open("min.pdb", 'w'))

    def md(self, heat_steps=10000, prod_steps=50000000, hw='cpu', log=True, save=False):

        from openmm import Platform, LangevinIntegrator, Vec3
        from openmm.app import Modeller, ForceField, \
            CutoffNonPeriodic, PME, Simulation, HBonds, NoCutoff, PDBFile
        from openmm.unit import angstrom, nanometer, picosecond, \
            kelvin, Quantity, molar, kilojoule_per_mole

        if log:
            LOGGER.info('Performing MD...')
        forcefield = ForceField(*self._forcefield)

        system = forcefield.createSystem(self._topology, 
                                         nonbondedMethod=NoCutoff,
                                         nonbondedCutoff=99.9*nanometer, 
                                         constraints=HBonds)
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 
                                        0.002*picosecond)

        if hw == 'cpu': 
            platform = Platform.getPlatformByName('CPU')
        elif hw == 'gpu':
            platform = Platform.getPlatformByName('CUDA')
        simulation = Simulation(self._topology, system, integrator, platform)
        simulation.context.setPositions(self._positions)
        simulation.step(heat_steps)
        simulation.step(prod_steps)
        self._positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        self._potential = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        if log:
            LOGGER.info('MD successfully completed.')
        if save:
            PDBFile.writeFile(self._topology, self._positions, 
                              open("md.pdb", 'w'))

    def build_anm(self, log=True):

        from openmm.unit import angstrom

        if log:
            LOGGER.info('Building ANM...')
        pos = self._positions[:self._topology.getNumAtoms()]

        tmp = self._atoms.copy()
        tmp.setCoords(pos)
        self._ca = tmp[self._idx_ca]
                            
        self._anm.buildHessian(self._ca)
        if log:
            LOGGER.info('ANM successfully built.')

    def sample(self, confs, **kwargs):

        tmp = []

        for conf in confs:
            ctmp = self._atoms.copy()
            ctmp.setCoords(conf)
            cg = ctmp[self._idx_ca]
                                                                                            
            self._positions=ctmp.getCoords()
                                                                                            
            self.build_anm(log=False)
                                                                                            
            self._anm.calcModes(self._n_modes)
                                                                                            
            anm_ex, atoms_all = extendModel(self._anm, cg, ctmp)
            ens_ex = sampleModes(anm_ex, atoms=ctmp, n_confs=self._n_confs, rmsd=self._rmsd)
            tmp.append(ens_ex.getCoordsets())

        tmp = [r for r in tmp if r is not None]

        confs_ex = np.concatenate(tmp)

        return confs_ex

    def hclust(self, confs):

        tmp3 = confs.reshape(-1, 3 * self._n_ca)
        rmsds = pdist(tmp3) / np.sqrt(self._n_ca)

        link = linkage(rmsds, method='average')
        labels = fcluster(link, t=self._maxclust,
                           criterion='maxclust') - 1

        nl = np.unique(labels)
        idx = OrderedDict()
        for i in nl:
            idx[i] = np.where(labels == i)[0]
                                                
        centers = np.empty(nl.size, dtype=int)
        for i in nl:
            tempc = confs[idx[i]]
            if tempc.shape[0] > 2:
                tmp = tempc.reshape(-1, 3 * self._n_ca)
                rmsds = pdist(tmp) / np.sqrt(self._n_ca)
                sim = np.exp(- squareform(rmsds) / rmsds.std())
                idx2 = sim.sum(1).argmax()
                centers[i] = idx[i][idx2]
            else:
                centers[i] = idx[i][0]

        return centers

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
                self.minimize(n_cycles=2, ref=False, log=False)
                target_ensemble.addCoordset(self._positions*10)

            if save:
                writePDB("anmd_mode_%s"%str(modeNum), target_ensemble)

        LOGGER.info('ANMD completed successfully.')

    def run_clustenm(self, n_gens=2, rmsd=1, n_cycles=2, heat_steps=10, prod_steps=10, sim=False, hw='cpu', save=True):

        if not sim:
            LOGGER.info('Starting classic CLUSTENM...')

        self._n_modes = 2
        self._n_confs = 2
        self._rmsd = rmsd
        cycle = 0
        conformer = self._positions
        potentials = [self._potential]
        new_shape = [1]
        for s in conformer.shape:
            new_shape.append(s)
        conf = conformer.reshape(new_shape)
        conformers = start_confs = conf
        keys = [(0, 0)]
                                                                                                  
        for i in range(1, n_gens+1):
            cycle += 1
            confs = self.sample(start_confs)
            chosen = self.hclust(confs[:, self._idx_ca])
            confs = confs[chosen]
            conf_list = []
            pot_list = []
            if sim:
                LOGGER.info('Minimizing confs and running MD in generation %s.'%cycle)
            else:
                LOGGER.info('Minimizing confs in generation %s.'%cycle)
            for conf in confs:
                self._positions = conf
                try:
                    self.minimize(n_cycles=n_cycles, hw=hw, ref=False, log=False)
                    if sim:
                        self.md(heat_steps=heat_steps, prod_steps=prod_steps, hw=hw, log=False, save=False)
                    conf_list.append(self._positions)
                    pot_list.append(self._potential)
                except:
                    LOGGER.info('OpenMM failure. The corresponding conf will be discarded.')
                    conf_list.append(self._positions)
                    pot_list.append(np.nan)

            idx = np.logical_not(np.isnan(pot_list))

            if not np.any(idx):
                LOGGER.info('All conformations were discarded. Exiting...')
                exit()

            pots = np.array(pot_list)[idx]
            confs = np.array(conf_list)[idx]
            potentials.extend(np.array(pot_list)[idx])

            tmp0 = self._atoms.copy()
            tmp0.setCoords(self._reference)
            n = confs[idx].shape[0]
            tmp1 = []
            for i in range(n):
                tmp2 = calcTransformation(confs[i, self._idx_ca],
                                          tmp0[self._idx_ca])
                tmp1.append(applyTransformation(tmp2, confs[i]))
                                                                  
            start_confs = np.array(tmp1)

            for j in range(start_confs.shape[0]):
                keys.append((i, j))
            conformers = np.vstack((conformers, start_confs))

        target_ensemble = Ensemble()
        target_ensemble.setAtoms(self._atoms)
        for conf in conformers:
            target_ensemble.addCoordset(conf*10)

        if save and sim:
            writePDB("clustenmd_ens.pdb", target_ensemble)
        elif save and not sim:
            writePDB("clustenm_ens.pdb", target_ensemble)

        if not sim:
            LOGGER.info('CLUSTENM completed successfully.')

    def run_clustenmd(self, n_gens=2, rmsd=1, n_cycles=2, hw='cpu', save=True):
        
        LOGGER.info('Starting classic CLUSTENMD...')

        self.run_clustenm(n_gens=n_gens, rmsd=rmsd, n_cycles=n_cycles, sim=True, hw=hw, save=save)
        
        LOGGER.info('CLUSTENMD completed successfully.')
