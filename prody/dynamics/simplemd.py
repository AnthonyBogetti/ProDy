from prody.proteins import parsePDB
from prody.utilities import LOGGER
import mdtraj as md
from openmm import Platform, LangevinIntegrator, Vec3
from openmm.app import PDBFile, Modeller, ForceField, CutoffNonPeriodic, PME, Simulation, HBonds, StateDataReporter, DCDReporter, PDBReporter, CheckpointReporter
from openmm.unit import angstrom, nanometers, picosecond, kelvin, Quantity, molar, kilojoule_per_mole, MOLAR_GAS_CONSTANT_R
from pdbfixer import PDBFixer
from sys import stdout

__all__ = ["SimpleMD"]

class SimpleMD():

    def __init__(self):

        self._pdbname = None
        self._ph = 7.0
        self._forcefield = ("amber99sbildn.xml", "amber99_obc.xml")
        self._temp = 303.15
        self._heat_steps = 1E4
        self._prod_steps = 5E5
        self._platform = "CUDA"
        self._topology = None
        self._positions = None
        self._simulation = None
        self._ready_sim = None
        self._extend = False
        self._extend_steps = 5E5
                                                                                                     
    def _fix(self):

        LOGGER.info("Fixing the structure.")
        parsePDB(self._pdbname, compressed=False)
        fixed = PDBFixer(self._pdbname+".pdb")
        fixed.missingResidues = {}
        fixed.findNonstandardResidues()
        fixed.replaceNonstandardResidues()
        fixed.removeHeterogens(False)
        fixed.findMissingAtoms()
        fixed.addMissingAtoms()
        fixed.addMissingHydrogens(self._ph)
        PDBFile.writeFile(fixed.topology, fixed.positions, open(self._pdbname+"_fixed.pdb", "w"))

        self._topology = fixed.topology
        self._positions = fixed.positions

    def _prep_sim(self):

        LOGGER.info("Building the system.")
        modeller = Modeller(self._topology, self._positions)
        forcefield = ForceField(*self._forcefield)
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
                                         nonbondedCutoff=1.0*nanometers, constraints=HBonds)
        integrator = LangevinIntegrator(self._temp*kelvin, 1/picosecond, 0.002*picosecond)
        platform = Platform.getPlatformByName(self._platform)
        simulation = Simulation(modeller.topology, system, integrator, platform)

        self._simulation = simulation

        simulation.context.setPositions(modeller.positions)

        LOGGER.info("Energy minimizing the system.")
        simulation.minimizeEnergy(tolerance=10*kilojoule_per_mole, maxIterations=0)

        LOGGER.info("Heating the system.")
        sdr = StateDataReporter("heat.log", 100, step=True, temperature=True)
        simulation.reporters.append(sdr)
        sdr._initializeConstants(simulation)
        temp = 0.0

        while temp < 303.5:
            simulation.step(1)
            ke = simulation.context.getState(getEnergy=True).getKineticEnergy()
            temp = (2 * ke / (sdr._dof * MOLAR_GAS_CONSTANT_R)).value_in_unit(kelvin)
        LOGGER.info("Heating is complete.")

        LOGGER.info("Equilibrating the system at the target temperature.")
        simulation.reporters.pop(-1)
        simulation.step(self._heat_steps)

        self._ready_sim = simulation

    def _run_sim(self):

        LOGGER.info("Running the simulation.")
        simulation = self._ready_sim
        simulation.reporters.append(StateDataReporter("prod.log", 50000, totalSteps=self._prod_steps, 
                                                      progress=True, speed=True))
        simulation.reporters.append(CheckpointReporter("snap.chk", 5000))
        simulation.reporters.append(PDBReporter("snap.pdb", 5000))
        simulation.reporters.append(DCDReporter("prod.dcd", 5000))
        simulation.step(self._prod_steps)
        LOGGER.info("The simulation is complete.")

    def _extend_sim(self):

       LOGGER.info("Extending the simulation.")
       simulation = self._simulation
       simulation.loadCheckpoint("snap.chk")
       simulation.reporters.append(StateDataReporter("extend.log", 50000, totalSteps=self._extend_steps,
                                                     progress=True, speed=True))
       simulation.reporters.append(CheckpointReporter("snap.chk", 5000))
       simulation.reporters.append(PDBReporter("snap.pdb", 5000))
       simulation.reporters.append(DCDReporter("extend.dcd", 5000))
       simulation.step(self._extend_steps)
       LOGGER.info("The extension is complete.")

    def run(self, pdbname, ph, forcefield, temp, 
            heat_steps, prod_steps, platform,
            extend, extend_steps, **kwargs):

        self._pdbname = pdbname
        self._ph = ph
        self._forcefield = forcefield
        self._temp = temp
        self._heat_steps = heat_steps
        self._prod_steps = prod_steps
        self._platform = platform
        self._extend = extend
        self._extend_steps = extend_steps

        if self._extend:
            self.extend_sim()
        else:
            self._fix()
            self._prep_sim()
            self._run_sim()
