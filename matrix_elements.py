import numpy as np

# All equations here refer to the Supplemental Material for PRL 122, 230401, 2019

# Computing two-body Hamiltonian matrix elements
# using com and com_distance as the global variables that they are
def two_body_h(mu_A, mu_B, r_AB):
        d_AB = np.linalg.norm(r_AB)
        n_AB = r_AB/d_AB # normal vector along A-B
        V_AB = np.inner(mu_A, mu_B) # mu_A . mu_B
        V_AB -= 3*np.inner(mu_A, n_AB)*np.inner(mu_B, n_AB) # -3*(mu_A.n_AB)(mu_B.n_AB)
        V_AB /= d_AB**3 # / |r_AB|**3
        return V_AB

# number of chromophores
n_chromo = 18

# read info from TeraChem outputs
gs_energies = np.zeros(n_chromo) # ground state monomer energy
es_energies = np.zeros(n_chromo) # excited state monomer energy
com = np.zeros([n_chromo, 3]) # coordinates of the center of mass
gs_dipole = np.zeros([n_chromo, 3]) # ground state dipole moment
es_dipole = np.zeros([n_chromo, 3]) # excited state dipole moment
t_dipole = np.zeros([n_chromo, 3]) # transition dipole moment
for outfile in range(n_chromo):

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

# Angstrom to bohr
com = com*1.8897161646320724

# Dipole momments for the same state are in D and transition dipole moments are in au
gs_dipole *= 0.393430307
es_dipole *= 0.393430307

# One-body Hamiltonian matrix elements
# S_A (Eq. 20)
# D_A (Eq. 21)
S_A = (gs_energies + es_energies)/2
D_A = (gs_energies - es_energies)/2

# Two-body Hamiltonian matrix elements
# (SA|SB) = (O_A 0_A + 1_A 1_A | O_B 0_B + 1_B 1_B)/4 (Eq. 28)
# (DA|SB) = (O_A 0_A - 1_A 1_A | O_B 0_B + 1_B 1_B)/4 (Eq. 30)
# (TA|SB) = (O_A 1_A | O_B 0_B + 1_B 1_B)/2 (Eq. 24)

# Diagonal elements of the Hamiltonian (Eq. 13)
# E = sum_A SA + sum_A>B (SA|SB)
E = np.sum(S_A)
for A in range(n_chromo - 1):
    mu_A = (gs_dipole[A, :] + es_dipole[A, :])/2
    for B in range(A + 1, n_chromo):
        mu_B = (gs_dipole[B, :] + es_dipole[B, :])/2
        E += two_body_h(mu_A, mu_B, com[A] - com[B])

# One-body Hamiltonian matrix elements
# Z_A = D_A + sum_B (DA|SB) (Eq. 14)
Z_A = np.zeros(n_chromo)
for A in range(n_chromo):
    mu_A = (gs_dipole[A, :] - es_dipole[A, :])/2
    Z_A[A] = D_A[A]
    for B in [x for x in range(n_chromo) if x != A]:
        mu_B = (gs_dipole[B, :] + es_dipole[B, :])/2
        Z_A[A] += two_body_h(mu_A, mu_B, com[A] - com[B])

# X_A = sum_B (TA|SB) (Eq. 15)
# X_A on the rhs is zero
X_A = np.zeros(n_chromo)
for A in range(n_chromo):
    mu_A = t_dipole[A, :]
    for B in [x for x in range(n_chromo) if x != A]:
        mu_B = (gs_dipole[B, :] + es_dipole[B, :])/2
        X_A[A] += two_body_h(mu_A, mu_B, com[A] - com[B])

# XX_AB = (TA|TB) = (0_A 1_A | O_B 1_B) (Eq. 23)
# XZ_AB = (TA|DB) = (0_A 1_A | O_B 0_B - 1_B 1_B)/2 (Eq. 25)
# ZX_AB = (DA|TB) = (O_A 0_A - 1_A 1_A | 0_B 1_B)/2 (Eq. 27)
# ZZ_AB = (DA|DB) = (O_A 0_A - 1_A 1_A | O_B 0_B - 1_B 1_B)/4 (Eq. 31)
XX_AB = np.zeros([n_chromo, n_chromo])
XZ_AB = np.zeros([n_chromo, n_chromo])
ZX_AB = np.zeros([n_chromo, n_chromo])
ZZ_AB = np.zeros([n_chromo, n_chromo])
for A in range(n_chromo - 1):
    for B in range(A + 1, n_chromo):
        XX_AB[A, B] = two_body_h(t_dipole[A, :], t_dipole[B, :], com[A] - com[B])
        XZ_AB[A, B] = two_body_h(t_dipole[A, :], (gs_dipole[B, :] - es_dipole[B, :])/2, com[A] - com[B])
        ZX_AB[A, B] = two_body_h((gs_dipole[A, :] - es_dipole[A, :])/2, t_dipole[B, :], com[A] - com[B])
        ZZ_AB[A, B] = two_body_h((gs_dipole[A, :] - es_dipole[A, :])/2, (gs_dipole[B, :] - es_dipole[B, :])/2, com[A] - com[B])

# Compute CIS matrix and diagonalize
cis_matrix = np.zeros([1 + n_chromo, 1 + n_chromo])

# H[0,0]
E_ref = E + np.sum(Z_A)
for A in range(n_chromo - 1):

    cis_matrix[A + 1, A + 1] = - 2.0 * Z_A[A]
    cis_matrix[A + 1, 0] = X_A[A]

    for B in range(A + 1, n_chromo):
        
        E_ref += ZZ_AB[A, B]
        cis_matrix[A + 1, A + 1] -= ZZ_AB[A, B] + ZZ_AB[B, A]
        cis_matrix[A + 1, 0] += 0.5 * (XZ_AB[A, B] + ZX_AB[B, A])
        cis_matrix[A + 1 , B + 1] = XX_AB[A, B]

    cis_matrix[0, A + 1] = cis_matrix[A + 1, 0] 
    cis_matrix[B + 1, A + 1] = cis_matrix[A + 1, B + 1]

cis_matrix[np.diag_indices_from(cis_matrix)] += E_ref
cis_energies, cis_states = np.linalg.eigh(cis_matrix)

# compute gate angles from CIS state vectors Eq. 60 of derivative preprint
# rows are angles and columns states
state_gate_angles = np.zeros([n_chromo, n_chromo + 1])
for state in range(1):
    for angle in range(n_chromo):
        state_gate_angles[angle, state] = np.arccos(cis_states[angle, state]/np.sqrt(np.sum(cis_states[angle:, state]**2)))
    state_gate_angles[-1, state] *= np.sign(cis_states[-1, state])

