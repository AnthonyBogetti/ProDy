'''
Copyright (c) 2025 Anthony Bogetti, Anupam Banerjee.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

__author__ = 'Anthony Bogetti'
__credits__ = ['Anupam Banerjee']
__email__ = ['anthony.bogetti@stonybrook.edu', 'anupam.banerjee@stonybrook.edu']

from itertools import product
from multiprocessing import cpu_count, Pool
from collections import OrderedDict
from os import chdir, mkdir
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
from .sampling import sampleModes
from prody.atomic import AtomGroup
from prody.measure import calcTransformation, applyTransformation, calcRMSD
from prody.ensemble import Ensemble
from prody.proteins import writePDB, parsePDB, writePDBStream, parsePDBStream
from prody.utilities import createStringIO, importLA, mad

la = importLA()
norm = la.norm

__all__ = ['HYBRID']

class HYBRID(Ensemble):

    '''
    HYBRID is a new interface for all hybrid ANM-based simulations in ProDy. PDBFixer and
    OpenMM are required.

    Instantiate a HYBRID object.
    '''

    def __init__(self, title=None):

        self._atoms = None
        self._nuc = None

        self._cutoff = 15.
        self._gamma = 1.
        self._n_modes = 3
        self._n_confs = 50
        self._rmsd = (0.,) 
        self._n_gens = 5

        self._maxclust = None
        self._threshold = None

        self._force_field = None
        self._maxIterations = 1000
        self._sim = True
        self._t_steps = None

        self._platform = None 
        self._parallel = False

        self._topology = None
        self._positions = None
        self._idx_cg = None
        self._n_cg = None
        self._cycle = 0
        self._time = 0
        self._indexer = None
        self._targeted = False
        self._tmdk = 10.

        self._cc = None

        super(HYBRID, self).__init__('Unknown')   # dummy title; will be replaced in the next line
        self._title = title

    def __getitem__(self, index):

        if isinstance(index, tuple):
            I = self._slice(index)
            if I is None:
                raise IndexError('index out of range %s' % str(index))
            index = I

        return super(HYBRID, self).__getitem__(index)

    def getAtoms(self, selected=True):

        'Returns atoms.'

        return super(HYBRID, self).getAtoms(selected)

    def _isBuilt(self):

        return self._confs is not None

    def setAtoms(self, atoms):

        '''
        Sets atoms.
        
        :arg atoms: *atoms* parsed by parsePDB

        '''

        atoms = atoms.select('not hetatm')

        self._nuc = atoms.select('nucleotide')

        if self._nuc is not None:

            idx_p = []
            for c in self._nuc.getChids():
                tmp = self._nuc[c].iterAtoms()
                for a in tmp:
                    if a.getName() in ['P', 'OP1', 'OP2', 'OP3']:
                        idx_p.append(a.getIndex())

            if idx_p:
                nsel = 'not index ' + ' '.join([str(i) for i in idx_p])
                atoms = atoms.select(nsel)

        if self._isBuilt():
            super(HYBRID, self).setAtoms(atoms)
        else:
            LOGGER.info('Fixing the structure ...')
            LOGGER.timeit('_hybrid_fix')
            self._fix(atoms)
            LOGGER.report('The structure was fixed in %.2fs.',
                          label='_hybrid_fix')

            if self._nuc is None:
                self._idx_cg = self._atoms.ca.getIndices()
                self._n_cg = self._atoms.ca.numAtoms()
            else:
                self._idx_cg = self._atoms.select("name CA C2 C4' P").getIndices()
                self._n_cg = self._atoms.select("name CA C2 C4' P").numAtoms()

            self._n_atoms = self._atoms.numAtoms()
            self._indices = None

    def getTitle(self):

        'Returns the title.'

        title = 'Unknown'
        if self._title is None:
            atoms = self.getAtoms()
            if atoms is not None:
                title = atoms.getTitle() + '_clustenm'
        else:
            title = self._title

        return title

    def setTitle(self, title):

        '''
        Set title.

        :arg title: Title of the HYBRID object.
        :type title: str 
        '''

        if not isinstance(title, str) and title is not None:
            raise TypeError('title must be either str or None')
        self._title = title

    def _fix(self, atoms):

        try:
            from pdbfixer import PDBFixer
            from openmm.app import PDBFile
        except ImportError:
            raise ImportError('Please install PDBFixer and OpenMM in order to use HYBRID.')

        stream = createStringIO()
        title = atoms.getTitle()
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
        fixed.addMissingHydrogens(7)

        stream = createStringIO()
        PDBFile.writeFile(fixed.topology, fixed.positions,
                          stream, keepIds=True)
        stream.seek(0)
        self._atoms = parsePDBStream(stream)
        self._atoms.setTitle(title)
        stream.close()

        self._topology = fixed.topology
        self._positions = fixed.positions

    def _prep_sim(self, coords, external_forces=[]):

        try:
            from openmm import Platform, LangevinIntegrator, Vec3
            from openmm.app import Modeller, ForceField, \
                CutoffNonPeriodic, PME, Simulation, HBonds
            from openmm.unit import angstrom, nanometers, picosecond, \
                kelvin, Quantity, molar
        except ImportError:
            raise ImportError('Please install PDBFixer and OpenMM in order to use HYBRID.')

        positions = Quantity([Vec3(*xyz) for xyz in coords], angstrom)
        modeller = Modeller(self._topology, positions)

        forcefield = ForceField(*self._force_field)

        system = forcefield.createSystem(modeller.topology,
                                         nonbondedMethod=CutoffNonPeriodic,
                                         nonbondedCutoff=99.9*nanometers,
                                         constraints=HBonds)

        for force in external_forces:
            system.addForce(force)

        integrator = LangevinIntegrator(300.*kelvin,
                                        1/picosecond,
                                        0.002*picosecond)

        # precision could be mixed, but single is okay.
        platform = self._platform if self._platform is None else Platform.getPlatformByName(self._platform)
        properties = None

        if self._platform is None:
            properties = {'Precision': 'single'}
        elif self._platform in ['CUDA', 'OpenCL']:
            properties = {'Precision': 'single'}

        simulation = Simulation(modeller.topology, system, integrator,
                                platform, properties)

        simulation.context.setPositions(modeller.positions)

        return simulation

    def _min_sim(self, coords):

        # coords: coordset   (numAtoms, 3) in Angstrom, which should be converted into nanometer

        try:
            from openmm.app import StateDataReporter
            from openmm.unit import kelvin, angstrom, kilojoule_per_mole, MOLAR_GAS_CONSTANT_R
        except ImportError:
            raise ImportError('Please install PDBFixer and OpenMM in order to use HYBRID.')

        simulation = self._prep_sim(coords=coords)

        # automatic conversion into nanometer will be carried out.
        # simulation.context.setPositions(coords * angstrom)

        try:
            simulation.minimizeEnergy(maxIterations=self._maxIterations)
            if self._sim:
                # heating-up the system
                sdr = StateDataReporter(stdout, 1, step=True, temperature=True)
                sdr._initializeConstants(simulation)

                simulation.step(self._t_steps[self._cycle])

            pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom)[:self._topology.getNumAtoms()]
            pot = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)

            return pot, pos

        except BaseException as be:
            LOGGER.warning('OpenMM exception: ' + be.__str__() + ' so the corresponding conformer will be discarded!')

            return np.nan, np.full_like(coords, np.nan)

    def _targeted_sim(self, coords0, coords1, tmdk=15., d_steps=100, n_max_steps=10000, ddtol=1e-3, n_conv=5):

        try:
            from openmm import CustomExternalForce
            from openmm.app import StateDataReporter
            from openmm.unit import nanometer, kelvin, angstrom, kilojoule_per_mole, MOLAR_GAS_CONSTANT_R
        except ImportError:
            raise ImportError('Please install PDBFixer and OpenMM in order to use HYBRID.')

        tmdk *= kilojoule_per_mole/angstrom**2
        tmdk = tmdk.value_in_unit(kilojoule_per_mole/nanometer**2)

        # coords1_ca = coords1[self._idx_cg, :]
        pos1 = coords1 * angstrom
        # pos1_ca = pos1[self._idx_cg, :]

        force = CustomExternalForce('tmdk*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
        force.addGlobalParameter('tmdk', 0.) 
        force.addPerParticleParameter('x0')
        force.addPerParticleParameter('y0')
        force.addPerParticleParameter('z0')
        force.setForceGroup(1)
        # for i, atm_idx in enumerate(self._idx_cg):
        #     pars = pos1_ca[i, :].value_in_unit(nanometer)
        #     force.addParticle(int(atm_idx), pars)

        n_atoms = coords0.shape[0]
        atom_indices = np.arange(n_atoms)
        for i, atm_idx in enumerate(atom_indices):
            pars = pos1[i, :].value_in_unit(nanometer)
            force.addParticle(int(atm_idx), pars)

        simulation = self._prep_sim([force])

        # automatic conversion into nanometer will be carried out.
        simulation.context.setPositions(coords0 * angstrom)

        dist = dist0 = calcRMSD(coords0, coords1)
        m_conv = 0
        n_steps = 0
        try:
            simulation.minimizeEnergy(maxIterations=self._maxIterations)

            # update parameters
            while n_steps < n_max_steps:
                simulation.context.setParameter('tmdk', tmdk)
                force.updateParametersInContext(simulation.context)

                simulation.step(d_steps)
                n_steps += d_steps

                # evaluate distance to destination
                pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom)
                d = calcRMSD(pos, coords1)
                dd = np.abs(dist - d)

                if dd < ddtol:
                    m_conv += 1

                if m_conv >= n_conv:
                    break

                dist = d

            LOGGER.debug('RMSD: %4.2f -> %4.2f' % (dist0, dist))

            simulation.context.setParameter('tmdk', 0.0)
            simulation.minimizeEnergy(maxIterations=self._maxIterations)

            pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom)
            pot = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)

            return pot, pos

        except BaseException as be:
            LOGGER.warning('OpenMM exception: ' + be.__str__() + ' so the corresponding conformer will be discarded!')

            return np.nan, np.full_like(coords0, np.nan)

    def _checkANM(self, anm):

        # use prody's ZERO parameter
        
        H = anm.getHessian()
        rank = np.linalg.matrix_rank(anm.getHessian(), tol=ZERO, hermitian=True)
        rank_diff = H.shape[0] - 6 - rank

        good = rank_diff <= 0   # '<' needed due to RTB

        if not good:
            # taking care cases with more than 6 zeros
            LOGGER.warn('Abnormal number of zero modes detected (%d detected, 6 expected), so this conformer is discarded!' % (6 + rank_diff))

        return good

    def _multi_targeted_sim(self, args):

        conf = args[0]
        coords = args[1]

        return self._targeted_sim(conf, coords, tmdk=self._tmdk)

    def _buildANM(self, cg):

        anm = ANM()
        anm.buildHessian(cg, cutoff=self._cutoff, gamma=self._gamma,
                         sparse=self._sparse, kdtree=self._kdtree,
                         nproc=self._nproc)

        return anm

    def _extendModel(self, model, nodes, atoms):

        if self._nuc is None:
            pass
        else:
            _, idx_n3, cnt = np.unique(nodes.nucleotide.getResindices(),
                                       return_index=True, return_counts=True)
            idx_c4p = np.where(nodes.getNames() == "C4'")[0]

            vpn3 = model.getEigvecs()

            for i, n, j in zip(idx_n3, cnt, idx_c4p):
                vpn3[3*i:3*(i+n)] = np.tile(vpn3[3*j:3*(j + 1), :], (n, 1))

            model.setEigens(vpn3, model.getEigvals())

        ext, _ = extendModel(model, nodes, atoms, norm=True)

        return ext

    def _sample(self, conf):

        tmp = self._atoms.copy()
        tmp.setCoords(conf)
        cg = tmp[self._idx_cg]

        anm_cg = self._buildANM(cg)

        n_confs = self._n_confs

        if not self._checkANM(anm_cg):
            return None

        anm_cg.calcModes(self._n_modes, turbo=self._turbo,
                         nproc=self._nproc)

        anm_ex = self._extendModel(anm_cg, cg, tmp)
        ens_ex = sampleModes(anm_ex, atoms=tmp,
                             n_confs=self._n_confs,
                             rmsd=self._rmsd[self._cycle])
        coordsets = ens_ex.getCoordsets()

        if self._fitmap is not None:
            LOGGER.info('Filtering for fitting in generation %d ...' % self._cycle)

            kept_coordsets = []
            if self._fitmap is not None:
                kept_coordsets.extend(self._filter(coordsets))
                n_confs = n_confs - len(kept_coordsets)

            if len(kept_coordsets) == 0:
                while len(kept_coordsets) == 0:
                    anm_ex = self._extendModel(anm_cg, cg, tmp)
                    ens_ex = sampleModes(anm_ex, atoms=tmp,
                                        n_confs=n_confs,
                                        rmsd=self._rmsd[self._cycle])
                    coordsets = ens_ex.getCoordsets()

                    if self._fitmap is not None:
                        kept_coordsets.extend(self._filter(coordsets))
                        n_confs = n_confs - len(kept_coordsets)

            if self._replace_filtered:
                while n_confs > 0: 
                    anm_ex = self._extendModel(anm_cg, cg, tmp)
                    ens_ex = sampleModes(anm_ex, atoms=tmp,
                                        n_confs=n_confs,
                                        rmsd=self._rmsd[self._cycle])
                    coordsets = ens_ex.getCoordsets()

                    if self._fitmap is not None:
                        kept_coordsets.extend(self._filter(coordsets))
                        n_confs = n_confs - len(kept_coordsets)

            coordsets = np.array(kept_coordsets)

        if self._targeted:
            if self._parallel:
                with Pool(cpu_count()) as p:
                    pot_conf = p.map(self._multi_targeted_sim,
                                     [(conf, coords) for coords in coordsets])
            else:
                pot_conf = [self._multi_targeted_sim((conf, coords)) for coords in coordsets]

            pots, poses = list(zip(*pot_conf))

            idx = np.logical_not(np.isnan(pots))
            coordsets = np.array(poses)[idx]

            LOGGER.debug('%d/%d sets of coordinates were moved to the target' % (len(poses), len(coordsets)))

        return coordsets

    def _rmsds(self, coords):

        # as long as there is no need for superposing conformations

        # coords: (n_conf, n_cg, 3)

        tmp = coords.reshape(-1, 3 * self._n_cg)

        return pdist(tmp) / np.sqrt(self._n_cg)

    def _hc(self, arg):

        # arg: coords   (n_conf, n_cg, 3)

        rmsds = self._rmsds(arg)
        # optimal_ordering=True can be slow, particularly on large datasets.
        link = linkage(rmsds, method='average')

        # fcluster gives cluster labels starting from 1

        if self._threshold is not None:
            hcl = fcluster(link, t=self._threshold[self._cycle],
                           criterion='distance') - 1

        if self._maxclust is not None:
            hcl = fcluster(link, t=self._maxclust[self._cycle],
                           criterion='maxclust') - 1

        return hcl

    def _centroid(self, arg):

        # arg: coords   (n_conf_clust, n_cg, 3)

        if arg.shape[0] > 2:
            rmsds = self._rmsds(arg)
            sim = np.exp(- squareform(rmsds) / rmsds.std())
            idx = sim.sum(1).argmax()
            return idx
        else:
            return 0   # or np.random.randint(low=0, high=arg.shape[0])

    def _centers(self, *args):

        # args[0]: coords   (n_conf, n_cg, 3)
        # args[1]: labels

        nl = np.unique(args[1])
        idx = OrderedDict()
        for i in nl:
            idx[i] = np.where(args[1] == i)[0]

        # Dictionary order is guaranteed to be insertion order by Python 3.7!
        wei = [idx[k].size for k in idx.keys()]
        centers = np.empty(nl.size, dtype=int)
        for i in nl:
            tmp = self._centroid(args[0][idx[i]])
            centers[i] = idx[i][tmp]

        return centers, wei
    
    def _filter(self, *args):

        ag = self._atoms.copy()
        confs = args[0]
        ccList = np.zeros(len(args[0]))
        for i in range(len(confs)-1,-1,-1):
            ag.setCoords(confs[i])
            sim_map = self._blurrer(ag.toTEMPyStructure(), self._fit_resolution, self._fitmap)
            cc = self._scorer.CCC(sim_map, self._fitmap)
            ccList[i] = cc
            if cc - self._cc_prev < 0:
                confs = np.delete(confs, i, 0)

        self._cc.extend(ccList)

        return confs

    def _generate(self, confs, **kwargs):

        LOGGER.info('Sampling conformers in generation %d ...' % self._cycle)
        LOGGER.timeit('_clustenm_gen')

        sample_method = self._sample

        if self._parallel:
            with Pool(cpu_count()) as p:
                tmp = p.map(sample_method, [conf for conf in confs])
        else:
            tmp = [sample_method(conf) for conf in confs]

        tmp = [r for r in tmp if r is not None]

        confs_ex = np.concatenate(tmp)

        confs_cg = confs_ex[:, self._idx_cg]

        LOGGER.info('Clustering in generation %d ...' % self._cycle)
        label_cg = self._hc(confs_cg)
        centers, wei = self._centers(confs_cg, label_cg)
        LOGGER.report('Centroids were generated in %.2fs.',
                      label='_clustenm_gen')
        
        confs_centers = confs_ex[centers]
        
        if self._fitmap is not None:
            self._cc_prev = max(self._cc)
            LOGGER.info('Best CC is %f from %d conformers' % (self._cc_prev, len(confs_cg)))

        if len(confs_cg) > 1:
            LOGGER.info('Clustering in generation %d ...' % self._cycle)
            label_cg = self._hc(confs_cg)
            centers, wei = self._centers(confs_cg, label_cg)
            LOGGER.report('Centroids were generated in %.2fs.',
                        label='_clustenm_gen')
            confs_centers = confs_ex[centers]
        else:
            confs_centers, wei = confs_cg, [len(confs_cg)]

        return confs_centers, wei

    def _superpose_cg(self, confs):
        tmp0 = self._getCoords()
        n = confs.shape[0]
        tmp1 = []
        for i in range(n):
            tmp2 = calcTransformation(confs[i, self._idx_cg],
                                      tmp0[self._idx_cg])
            tmp1.append(applyTransformation(tmp2, confs[i]))

        return np.array(tmp1)

    def _build(self, conformers, keys, potentials, sizes):

        self.addCoordset(conformers)
        self.setData('size', sizes)
        self.setData('key', keys)
        self.setData('potential', potentials)

    def addCoordset(self, coords):

        '''
        Add coordinate set(s) to the ensemble.

        :arg coords: coordinate  set(s)
        :type coords: :class:`~numpy.ndarray`
        '''

        self._indexer = None
        super(HYBRID, self).addCoordset(coords)

    def getData(self, key, gen=None):

        '''
        Returns data.

        :arg key: Key
        :type key: str

        :arg gen: Generation
        :type gen: int
        '''

        keys = super(HYBRID, self)._getData('key')
        data = super(HYBRID, self).getData(key)

        if gen is not None:
            data_ = []
            for k, d in zip(keys, data):
                g, _ = k
                if g == gen:
                    data_.append(d)
            data = np.array(data_)
        return data

    def getKeys(self, gen=None):

        '''
        Returns keys.

        :arg gen: Generation number.
        :type gen: int
        '''

        return self.getData('key', gen)

    def getLabels(self, gen=None):

        '''
        Returns labels.

        :arg gen: Generation number.
        :type gen: int
        '''

        keys = self.getKeys(gen)
        labels = ['%d_%d' % tuple(k) for k in keys]

        return labels

    def getPotentials(self, gen=None):

        '''
        Returns potentials.

        :arg gen: Generation number.
        :type gen: int
        '''

        return self.getData('potential', gen)

    def getSizes(self, gen=None):

        '''
        Returns the number of unminimized conformers represented by a cluster centroid.

        :arg gen: Generation number.
        :type gen: int
        '''

        return self.getData('size', gen)

    def numGenerations(self):

        'Returns the number of generations.'

        return self._n_gens

    def numConfs(self, gen=None):

        '''
        Returns the number of conformers.

        :arg gen: Generation number.
        :type gen: int
        '''

        if gen is None:
            return super(HYBRID, self).numConfs()

        keys = self._getData('key')
        n_confs = 0
        for g, _ in keys:
            if g == gen:
                n_confs += 1

        return n_confs

    def _slice(self, indices):

        if len(indices) == 0:
            raise ValueError('indices (tuple) cannot be empty')

        if self._indexer is None:
            keys = self._getData('key')
            entries = [[] for _ in range(self.numGenerations() + 1)]
            for i, (gen, _) in enumerate(keys):
                entries[gen].append(i)

            n_conf_per_gen = np.max([len(entry) for entry in entries])
            for entry in entries:
                for i in range(len(entry), n_conf_per_gen):
                    entry.append(-1)

            indexer = self._indexer = np.array(entries)
        else:
            indexer = self._indexer
        
        full_serials = indexer[indices]

        if np.isscalar(full_serials):
            index = full_serials
            indices = None if index == -1 else index
        else:
            full_serials = full_serials.flatten()
            indices = []
            for s in full_serials:
                if s != -1:
                    indices.append(s)
            indices = np.array(indices) if indices else None

        return indices

    def _getCoordsets(self, indices=None, selected=True):

        '''
        Returns the coordinate set(s) at given *indices*, which may be
        an integer, a list of integers, a tuple of (generation, index), or **None**. 
        **None** returns all coordinate sets. For reference coordinates, use :meth:`getCoords`
        method.
        '''

        if isinstance(indices, tuple):
            I = self._slice(indices)
            if I is None:
                raise IndexError('index out of range %s' % str(indices))
        else:
            I = indices

        return super(HYBRID, self)._getCoordsets(I, selected)

    def writePDBFixed(self):

        'Write the fixed (initial) structure to a pdb file.'

        try:
            from openmm.app import PDBFile
        except ImportError:
            raise ImportError('Please install PDBFixer and OpenMM in order to use HYBRID.')

        PDBFile.writeFile(self._topology,
                          self._positions,
                          open(self.getTitle()[:-8] + 'fixed.pdb', 'w'),
                          keepIds=True)

    def writePDB(self, filename=None, single=True, **kwargs):

        '''
        Write conformers in PDB format to a file.
        
        :arg filename: The name of the file. If it is None (default), the title of the HYBRID  will be used.
        :type filename: str

        :arg single: If it is True (default), then the conformers will be saved into a single PDB file with
            each conformer as a model. Otherwise, a directory will be created with the filename,
            and each conformer will be saved as a separate PDB file.
        :type single: bool
        '''

        if filename is None:
            filename = self.getTitle()

        if single:
            filename = writePDB(filename, self)
            LOGGER.info('PDB file saved as %s' % filename)
        else:
            direc = filename
            if isdir(direc):
                LOGGER.warn('%s is not empty; will be flooded' % direc)
            else:
                mkdir(direc)

            LOGGER.info('Saving files ...')
            for i, lab in enumerate(self.getLabels()):
                filename = '%s/%s'%(direc, lab)
                writePDB(filename, self, csets=i)
            LOGGER.info('PDB files saved in %s ...'%direc)

    def run(self, cutoff=15., n_modes=3, gamma=1., n_confs=50, rmsd=1.0,
            n_gens=5, maxclust=None, threshold=None,
            sim=True, force_field=None,
            t_steps_i=1000, t_steps_g=7500, **kwargs):

        '''
        Performs a HYBRID simulation.

        :arg cutoff: Cutoff distance (A) for pairwise interactions used in ANM
            computations, default is 15.0 A.
        :type cutoff: float

        :arg gamma: Spring constant of ANM, default is 1.0.
        :type gamma: float

        :arg n_modes: Number of non-zero eigenvalues/vectors to calculate.
        :type n_modes: int

        :arg n_confs: Number of new conformers to be generated based on any conformer
            from the previous generation, default is 50.
        :type n_confs: int
            
        :arg rmsd: Average RMSD of the new conformers with respect to the conformer
            from which they are generated, default is 1.0 A.
            A tuple of floats can be given, e.g. (1.0, 1.5, 1.5) for subsequent generations.
            Note: In the case of ClustENMv1, this value is the maximum rmsd, not the average.
        :type rmsd: float, tuple of floats

        :arg n_gens: Number of generations.
        :type n_gens: int

        :arg maxclust: Maximum number of clusters for each generation, default in None.
            A tuple of integers can be given, e.g. (10, 30, 50) for subsequent generations.
            Warning: Either maxclust or RMSD threshold should be given! For large number of
            generations and/or structures, specifying maxclust is more efficient.
        :type maxclust: int or tuple of integers

        :arg threshold: RMSD threshold to apply when forming clusters, default is None.
            This parameter has been used in ClustENMv1, setting it to 75% of the maximum RMSD
            value used for sampling. A tuple of floats can be given, e.g. (1.5, 2.0, 2.5)
            for subsequent generations.
            Warning: This threshold should be chosen carefully in ClustENMv2 for efficiency.
        :type threshold: float or tuple of floats

        :arg force_field: Implicit solvent force field is ('amber14-all.xml', 'implicit/gbn2.xml'). 
        :type force_field: tuple of strings
        
        :arg maxIterations: Maximum number of iterations to perform during energy minimization.
            If this is 0 (default), minimization is continued until the results converge without
            regard to how many iterations it takes.
        :type maxIterations: int

        :arg sim: If it is True (default), a short MD simulation is performed after energy minimization.
        :type sim: bool

        :arg t_steps_i: Duration of MD simulation (number of time steps) for the starting structure
            following the heating-up phase, default is 1000. Each time step is 2.0 fs.
            Note: Default value reduces possible drift from the starting structure. 
        :type t_steps_i : int

        :arg t_steps_g: Duration of MD simulations (number of time steps) to run for each conformer
            following the heating-up phase, default is 7500. Each time step is 2.0 fs.
            A tuple of integers can be given, e.g. (3000, 5000, 7000) for subsequent generations.
        :type t_steps_g: int or tuple of integers

        :arg platform: Architecture on which the OpenMM part runs, default is None.
            It can be chosen as 'CUDA', 'OpenCL' or 'CPU'.
            For efficiency, 'CUDA' or 'OpenCL' is recommended.
        :type platform: str

        :arg parallel: If it is True (default is False), conformer generation will be parallelized.
        :type parallel: bool

        :arg fitmap: Cryo-EM map for fitting using a protocol similar to MDeNM-EMFit
            Default *None*
        :type fitmap: EMDMAP

        :arg fit_resolution: Resolution for comparing to cryo-EM map for fitting
            Default 5 Angstroms
        :type fit_resolution: float

        :arg replace_filtered: If it is True (default is False), conformer sampling and filtering 
            will be repeated until the desired number of conformers have been kept.
        :type replace_filtered: bool  
        '''

        if self._isBuilt():
            raise ValueError('HYBRID ensemble has been built; please start a new instance')

        # set up parameters
        self._cutoff = cutoff
        self._n_modes = n_modes
        self._gamma = gamma
        self._sparse = kwargs.get('sparse', False)
        self._kdtree = kwargs.get('kdtree', False)
        self._turbo = kwargs.get('turbo', False)
        self._nproc = kwargs.pop('nproc', 0)
        if kwargs.get('zeros', False):
            LOGGER.warn('HYBRID cannot use zero modes so ignoring this kwarg')

        self._n_confs = n_confs
        self._rmsd = (0.,) + rmsd if isinstance(rmsd, tuple) else (0.,) + (rmsd,) * n_gens
        self._n_gens = n_gens
        self._platform = kwargs.pop('platform', None)
        self._parallel = kwargs.pop('parallel', False)
        self._targeted = kwargs.pop('targeted', False)
        self._tmdk = kwargs.pop('tmdk', 15.)

        self._fitmap = kwargs.pop('fitmap', None)
        if self._fitmap is not None:
            try:
                from TEMPy.protein.structure_blurrer import StructureBlurrer
                from TEMPy.protein.scoring_functions import ScoringFunctions
            except ImportError:
                LOGGER.warn('TEMPy must be installed to use fitmap so ignoring this kwarg')
                self._fitmap = None
        
        if self._fitmap is not None:
            self._fitmap = self._fitmap.toTEMPyMap()
            self._fit_resolution = kwargs.get('fit_resolution', 5)
            self._replace_filtered = kwargs.pop('replace_filtered', False)

        if maxclust is None and threshold is None and n_gens > 0:
            raise ValueError('Either maxclust or threshold should be set!')
        
        if maxclust is None:
            self._maxclust = None
        else:
            if isinstance(maxclust, tuple):
                self._maxclust = (0,) + maxclust
            else:
                self._maxclust = (0,) + (maxclust,) * n_gens

            if len(self._maxclust) != self._n_gens + 1:
                raise ValueError('size mismatch: %d generations were set; %d maxclusts were given' % (self._n_gens + 1, self._maxclust))

        if threshold is None:
            self._threshold = None
        else:
            if isinstance(threshold, tuple):
                self._threshold = (0,) + threshold
            else:
                self._threshold = (0,) + (threshold,) * n_gens

            if len(self._threshold) != self._n_gens + 1:
                raise ValueError('size mismatch: %d generations were set; %d thresholds were given' % (self._n_gens + 1, self._threshold))

        self._force_field = ('amber14-all.xml', 'implicit/gbn2.xml') if force_field is None else force_field
        self._maxIterations = kwargs.pop('maxIterations', 1000)
        self._sim = sim

        if self._sim:
            if isinstance(t_steps_g, tuple):
                self._t_steps = (t_steps_i,) + t_steps_g
            else:
                self._t_steps = (t_steps_i,) + (t_steps_g,) * n_gens

        self._cycle = 0

        # check for discontinuity in the structure
        gnm = GNM()
        gnm.buildKirchhoff(self._atoms[self._idx_cg], cutoff=self._cutoff)
        K = gnm.getKirchhoff()
        rank_diff = (len(K) - 1
                     - np.linalg.matrix_rank(K, tol=ZERO, hermitian=True))
        if rank_diff != 0:
            raise ValueError('atoms has disconnected parts; please check the structure')

        if self._fitmap is not None:
            self._blurrer = StructureBlurrer().gaussian_blur_real_space
            sim_map_start = self._blurrer(self._atoms.toTEMPyStructure(),
                                          self._fit_resolution,
                                          self._fitmap)
            self._scorer = ScoringFunctions()
            self._cc_prev = self._scorer.CCC(self._fitmap, sim_map_start)
            self._cc = [self._cc_prev]
            LOGGER.info('Starting CC is %f' % self._cc_prev)

        LOGGER.timeit('_clustenm_overall')

        LOGGER.info('Generation 0 ...')

        if self._sim:
            if self._t_steps[0] != 0:
                LOGGER.info('Minimization, heating-up & simulation in generation 0 ...')
            else:
                LOGGER.info('Minimization & heating-up in generation 0 ...')
        else:
            LOGGER.info('Minimization in generation 0 ...')
        LOGGER.timeit('_clustenm_min')
        potential, conformer = self._min_sim(self._atoms.getCoords())
        if np.isnan(potential):
            raise ValueError('Initial structure could not be minimized. Try again and/or check your structure.')

        LOGGER.report(label='_clustenm_min')

        LOGGER.info('#' + '-' * 19 + '/*\\' + '-' * 19 + '#')

        self.setCoords(conformer)

        potentials = [potential]
        sizes = [1]
        new_shape = [1]
        for s in conformer.shape:
            new_shape.append(s)
        conf = conformer.reshape(new_shape)
        conformers = start_confs = conf
        keys = [(0, 0)]

        for i in range(1, self._n_gens+1):
            self._cycle += 1
            LOGGER.info('Generation %d ...' % i)
            confs, weights = self._generate(start_confs)
            if self._sim:
                if self._t_steps[i] != 0:
                    LOGGER.info('Minimization, heating-up & simulation in generation %d ...' % i)
                else:
                    LOGGER.info('Minimization & heating-up in generation %d ...' % i)
            else:
                LOGGER.info('Minimization in generation %d ...' % i)
            LOGGER.timeit('_clustenm_min_sim')

            pot_conf = [self._min_sim(conf) for conf in confs]

            LOGGER.report('Structures were sampled in %.2fs.',
                          label='_clustenm_min_sim')
            LOGGER.info('#' + '-' * 19 + '/*\\' + '-' * 19 + '#')

            pots, confs = list(zip(*pot_conf))
            idx = np.logical_not(np.isnan(pots))
            weights = np.array(weights)[idx]
            pots = np.array(pots)[idx]
            confs = np.array(confs)[idx]

            idx = np.full(pots.size, True, dtype=bool)

            sizes.extend(weights[idx])
            potentials.extend(pots[idx])
            start_confs = self._superpose_cg(confs[idx])

            for j in range(start_confs.shape[0]):
                keys.append((i, j))
            conformers = np.vstack((conformers, start_confs))

        LOGGER.timeit('_clustenm_ens')
        LOGGER.info('Creating an ensemble of conformers ...')

        self._build(conformers, keys, potentials, sizes)
        LOGGER.report('Ensemble was created in %.2fs.', label='_clustenm_ens')

        self._time = LOGGER.timing(label='_clustenm_overall')
        LOGGER.report('All completed in %.2fs.', label='_clustenm_overall')

    def writeParameters(self, filename=None):

        '''
        Write the parameters defined to a text file.

        :arg filename: The name of the file. If it is None (default), the title of the HYBRID will be used.
        :type filename: str
        '''

        title = self.getTitle()
        if filename is None:
            filename = '%s_parameters.txt' % title

        with open(filename, 'w') as f:
            f.write('title = %s\n' % title)
            f.write('cutoff = %4.2f A\n' % self._cutoff)
            f.write('n_modes = %d\n' % self._n_modes)
            f.write('rmsd = (%s)\n' % ', '.join([str(i) + ' A' for i in self._rmsd[1:]]))
            f.write('n_gens = %d\n' % self._n_gens)
            if self._threshold is not None:
                f.write('threshold = %s\n' % str(self._threshold[1:]))
            if self._maxclust is not None:
                f.write('maxclust = %s\n' % str(self._maxclust[1:]))
            f.write('force_field = (%s, %s)\n' % self._force_field)
            if self._maxIterations != 0:
                f.write('maxIteration = %d\n' % self._maxIterations)
            if self._sim:
                f.write('t_steps = %s\n' % str(self._t_steps))
            if self._platform is not None:
                f.write('platform = %s\n' % self._platform)
            else:
                f.write('platform = Default\n')
            if self._parallel:
                f.write('parallel = %s\n' % self._parallel)

            f.write('total time = %4.2f s' % self._time)


class ClustRTB(HYBRID):

    'Experimental.'

    def __init__(self, title=None):
        super(ClustRTB, self).__init__(title)
        self._blocks = None
        self._scale = 64.
        self._h = 100.

    def _buildANM(self, ca):
        blocks = self._blocks
        anm = RTB()
        anm.buildHessian(ca, blocks, cutoff=self._cutoff, gamma=self._gamma)

        return anm

    def setBlocks(self, blocks):
        self._blocks = blocks

    def run(self, **kwargs):
        if self._blocks is None:
            raise ValueError('blocks are not set')

        super(ClustRTB, self).run(**kwargs)


class ClustImANM(HYBRID):

    'Experimental.'

    def __init__(self, title=None):
        super(ClustImANM, self).__init__(title)
        self._blocks = None
        self._scale = 64.
        self._h = 100.

    def _buildANM(self, ca):
        blocks = self._blocks
        anm = imANM()
        anm.buildHessian(ca, blocks, cutoff=self._cutoff, 
                         gamma=self._gamma, scale=self._scale,
                         h=self._h)

        return anm

    def setBlocks(self, blocks):
        self._blocks = blocks

    def run(self, **kwargs):
        self._scale = kwargs.pop('scale', 64.)
        self._h = kwargs.pop('h', 100.)
        if self._blocks is None:
            raise ValueError('blocks are not set')

        super(ClustImANM, self).run(**kwargs)


class ClustExANM(HYBRID):

    'Experimental.'

    def _buildANM(self, ca):
        anm = exANM()
        anm.buildHessian(ca, cutoff=self._cutoff, gamma=self._gamma, R=self._R,
                         Ri=self._Ri, r=self._r, h=self._h, exr=self._exr,
                         gamma_memb=self._gamma_memb, hull=self._hull, lat=self._lat,
                         center=self._centering)

        return anm

    def run(self, **kwargs):
        depth = kwargs.pop('depth', None)
        h = depth / 2 if depth is not None else None
        self._h = kwargs.pop('h', h)
        self._R = float(kwargs.pop('R', 80.))
        self._Ri = float(kwargs.pop('Ri', 0.))
        self._r = float(kwargs.pop('r', 3.1))
        self._lat = str(kwargs.pop('lat', 'FCC'))
        self._exr = float(kwargs.pop('exr', 5.))
        self._hull = kwargs.pop('hull', True)
        self._centering = kwargs.pop('center', True)
        self._turbo = kwargs.pop('turbo', True)
        self._gamma_memb = kwargs.pop('gamma_memb', 1.)

        super(ClustExANM, self).run(**kwargs)
