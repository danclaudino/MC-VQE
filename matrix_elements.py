import numpy as np

# Computing two-body Hamiltonian matrix elements
# using com and com_distance as the global variables that they are
def two_body_hamiltonian(mu_A, mu_B):
    V_AB = np.zeros(mu_A.shape[0])
    for i in range(mu_A.shape[0]):
        n_AB = (com[i-1]-com[i])/np.linalg.norm(com[i-1]-com[i]) # normal vector along A-B
        V_AB[i] = np.inner(mu_A[i-1, :], mu_B[i, :]) # mu_A . mu_B
        V_AB[i] -= 3*np.inner(mu_A[i-1, :], n_AB)*np.inner(mu_B[i, :], n_AB) # -3*(mu_A.n_AB)(mu_B.n_AB)
        V_AB[i] /= com_distance[i-1,i]**3 # /r_AB**3
    return V_AB

# read info from TeraChem outputs
gs_energies = np.zeros(18) # ground state monomer energy
es_energies = np.zeros(18) # excited state monomer energy
com = np.zeros([18, 3]) # coordinates of the center of mass
gs_dipole = np.zeros([18, 3]) # ground state dipole moment
es_dipole = np.zeros([18, 3]) # excited state dipole moment
t_dipole = np.zeros([18, 3]) # transition dipole moment

for outfile in range(18):

    filename = '../packet/classical/' + str(outfile + 1) + '/out'
    with open(filename, 'r') as f:
        content = f.readlines()

    line = 0
    while line < len(content):
        if content[line][:12] == 'FINAL ENERGY':
            gs_energies[outfile] = float(content[line][13:29])
            line += 1
            com[outfile] = np.array([float(x) for x in content[line].split('{')[1].split('}')[0].split(',')])
            line += 1
            gs_dipole[outfile] = np.array([float(x) for x in content[line].split('{')[1].split('}')[0].split(',')])
            line += 1
        if content[line][:9] == 'Unrelaxed':
            es_energies[outfile] = float(content[line-3][10:30])
            line += 4
            es_dipole[outfile] = np.array([float(x) for x in content[line][9:42].split(' ') if x != ''])
            line += 7
            t_dipole[outfile] = np.array([float(x) for x in content[line][9:42].split(' ') if x != ''])
        else:
            line += 1
# end of TeraChem output reading

# COM matrix
com_distance = np.zeros([18,18])
for i in range(17):
    for j in range(i+1, 18):
        com_distance[i, j] = np.linalg.norm((com[i] - com[j])*1.8897161646320724)
        com_distance[j, i] = np.linalg.norm((com[i] - com[j])*1.8897161646320724)

# Dipole momment for the same state are in D and transition dipole moments are in au
gs_dipole *= 0.393430307
es_dipole *= 0.393430307

ref = np.load('../packet/hamiltonian/exciton.npz')
r = ref['R']
x = ref['X']
y = ref['Y']
z = ref['Z']

# One-body Hamiltonian matrix elements
# S_A (Eq. 20), D_A (Eq. 21)
S_A = (gs_energies + es_energies)/2
D_A = (gs_energies - es_energies)/2

# XX_AB = (TA|TB) = (0_A 1_A | O_B 1_B) (Eq. 23)
# XZ_AB = (TA|DB) = (0_A 1_A | O_B 0_B - 1_B 1_B)/2 (Eq. 25)
# ZZ_AB = (DA|TB) = (O_A 0_A - 1_A 1_A | 0_B 1_B)/2 (Eq. 25)
XX = two_body_hamiltonian(t_dipole, t_dipole)
XZ = two_body_hamiltonian(t_dipole, (gs_dipole - es_dipole)/2)
ZX = two_body_hamiltonian((gs_dipole - es_dipole)/2, t_dipole)
ZZ = two_body_hamiltonian((gs_dipole - es_dipole)/2, (gs_dipole - es_dipole)/2)