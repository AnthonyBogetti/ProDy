import mdtraj as md

traj = md.load("1ake_H_amber.pdb")
print(md.compute_dihedrals(traj, [[0,1,11,4], [19,20,31,23]]))
