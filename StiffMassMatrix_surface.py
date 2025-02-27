# this script takes surface mesh as input
# and it's calculate stiffness and mass matrices for 3D surface mesh with setted up thickness
# if you suppose that your material is filled 3D geometry - use StMassMatrix.py for correct calculation
import numpy as np
import meshio
from scipy.sparse import lil_matrix, csr_matrix

mesh = meshio.read("3d_models/your_mesh.msh")  # not obj and not stl

nodes = mesh.points

if "triangle" not in mesh.cells_dict:
    raise ValueError("Mesh does not contain triangular elements! Required for 2D FEM.")

elements = mesh.cells_dict["triangle"]
n_nodes = len(nodes)

print(f"Loaded mesh with {n_nodes} nodes and {len(elements)} triangular elements.")

K = lil_matrix((2 * n_nodes, 2 * n_nodes))
M = lil_matrix((2 * n_nodes, 2 * n_nodes))

E = 210e9  # Young's modulus (Pa)
nu = 0.3   # Poisson's ratio
rho = 7800  # Density (kg/m³)
t = 0.01  # Thickness of the 2D surface

# Elasticity matrix (D-matrix) for 2D plane stress
D = (E / (1 - nu**2)) * np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1 - nu) / 2]
])

for element in elements:
    n1, n2, n3 = element
    Xe = np.array([nodes[n1], nodes[n2], nodes[n3]])

    # area of the triangle for FEM
    X1, X2, X3 = Xe
    A = 0.5 * abs(X1[0] * (X2[1] - X3[1]) + X2[0] * (X3[1] - X1[1]) + X3[0] * (X1[1] - X2[1]))

    if A <= 0:
        raise ValueError(f"Invalid or degenerate triangle detected! Element {element}")

    # strain-displacement matrix
    B = np.zeros((3, 6))
    B[0, 0] = X2[1] - X3[1]
    B[0, 2] = X3[1] - X1[1]
    B[0, 4] = X1[1] - X2[1]

    B[1, 1] = X3[0] - X2[0]
    B[1, 3] = X1[0] - X3[0]
    B[1, 5] = X2[0] - X1[0]

    B[2, 0] = X3[0] - X2[0]
    B[2, 1] = X2[1] - X3[1]
    B[2, 2] = X1[0] - X3[0]
    B[2, 3] = X3[1] - X1[1]
    B[2, 4] = X2[0] - X1[0]
    B[2, 5] = X1[1] - X2[1]

    B /= (2 * A)

    # Ke = t * A * (B^T * D * B)
    Ke = t * A * (B.T @ D @ B)

    dofs = np.array([2 * n1, 2 * n1 + 1,
                     2 * n2, 2 * n2 + 1,
                     2 * n3, 2 * n3 + 1])

    for i in range(6):
        for j in range(6):
            K[dofs[i], dofs[j]] += Ke[i, j]

    Me = np.identity(6) * (rho * t * A / 3)

    for i in range(6):
        M[dofs[i], dofs[i]] += Me[i, i]

K = K.tocsr()
M = M.tocsr()

print(f"Stiffness Matrix Shape: {K.shape}")
print(f"Mass Matrix Shape: {M.shape}")
