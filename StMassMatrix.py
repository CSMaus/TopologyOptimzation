# calculate stiffness and mass matrices for 3D Geometry mesh (i e filled volume)

import numpy as np
import meshio
from scipy.sparse import lil_matrix, csr_matrix

mesh = meshio.read("3d_models/untitled.msh")
isGeometryMesh = True

nodes = mesh.points

if "tetra" in mesh.cells_dict:
    elements = mesh.cells_dict["tetra"]
elif "triangle" in mesh.cells_dict:
    print("Warning: Mesh contains only surface triangles, not volume tetrahedra.")
    elements = mesh.cells_dict["triangle"]
    isGeometryMesh = False
else:
    raise ValueError("Unsupported mesh format: No tetrahedral elements found.")

print(f"Loaded mesh: {len(nodes)} nodes, {len(elements)} elements")

n_nodes = len(nodes)

print(f"Loaded mesh with {n_nodes} nodes and {len(elements)} tetrahedral elements.")

K = lil_matrix((3 * n_nodes, 3 * n_nodes))
M = lil_matrix((3 * n_nodes, 3 * n_nodes))

E = 210e9  # Young's modulus (Pa)
nu = 0.3  # Poisson's ratio
rho = 7800  # Density (kg/m³)

# Lame parameters
lambda_lame = (E * nu) / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# Elasticity tensor (D-matrix)
D = np.array([
    [1, nu, nu, 0, 0, 0],
    [nu, 1, nu, 0, 0, 0],
    [nu, nu, 1, 0, 0, 0],
    [0, 0, 0, (1 - nu) / 2, 0, 0],
    [0, 0, 0, 0, (1 - nu) / 2, 0],
    [0, 0, 0, 0, 0, (1 - nu) / 2]
]) * (E / ((1 + nu) * (1 - 2 * nu)))

# element-wise stiffness and mass matrices
for element in elements:
    n1, n2, n3, n4 = element
    Xe = np.array([nodes[n1], nodes[n2], nodes[n3], nodes[n4]])

    #  Jacobian matrix and volume
    X1, X2, X3, X4 = Xe
    J = np.array([
        [X2[0] - X1[0], X3[0] - X1[0], X4[0] - X1[0]],
        [X2[1] - X1[1], X3[1] - X1[1], X4[1] - X1[1]],
        [X2[2] - X1[2], X3[2] - X1[2], X4[2] - X1[2]]
    ])

    detJ = np.linalg.det(J)
    if detJ <= 0:
        raise ValueError(f"Invalid or inverted tetrahedron detected! Element {element}")

    V = abs(detJ) / 6.0  # Tetrahedral volume

    #inverse of the Jacobian
    J_inv = np.linalg.inv(J)

    # strain-displacement matrix (B-matrix)
    grad_N = np.array([
        [-1, -1, -1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]) @ J_inv

    B = np.zeros((6, 12))
    B[0, 0::3] = grad_N[:, 0]
    B[1, 1::3] = grad_N[:, 1]
    B[2, 2::3] = grad_N[:, 2]
    B[3, 0::3] = grad_N[:, 1]
    B[3, 1::3] = grad_N[:, 0]
    B[4, 1::3] = grad_N[:, 2]
    B[4, 2::3] = grad_N[:, 1]
    B[5, 0::3] = grad_N[:, 2]
    B[5, 2::3] = grad_N[:, 0]

    # element stiffness matrix: Ke = B^T * D * B * V
    Ke = (B.T @ D @ B) * V

    # Assemble global stiffness matrix
    dofs = np.array([3 * n1, 3 * n1 + 1, 3 * n1 + 2,
                     3 * n2, 3 * n2 + 1, 3 * n2 + 2,
                     3 * n3, 3 * n3 + 1, 3 * n3 + 2,
                     3 * n4, 3 * n4 + 1, 3 * n4 + 2])

    for i in range(12):
        for j in range(12):
            K[dofs[i], dofs[j]] += Ke[i, j]

    # lumped mass matrix (M = (rho * V / 4) * I)
    Me = np.identity(12) * (rho * V / 4)

    # global mass matrix
    for i in range(12):
        M[dofs[i], dofs[i]] += Me[i, i]

# CSR format for efficiency (?)
# Todo: adapt for topology optimization code
K = K.tocsr()
M = M.tocsr()

print(f"Stiffness Matrix Shape: {K.shape}")
print(f"Mass Matrix Shape: {M.shape}")


