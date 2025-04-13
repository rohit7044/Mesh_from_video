import trimesh
import numpy as np


# Extract meanshape from Face_model.npy
face_model = np.load('assets/Face_model.npy',allow_pickle=True).item()

#Get Vertices of Mean Shape
mean_shape_verts = face_model['u'].squeeze().reshape(-1, 3)

#Get triangles of Mean Shape
mean_shape_faces = face_model['tri'].astype(np.int32)
# print(mean_shape_faces.shape)

#Visualize Mesh ---Won't work for WSL
mean_shape_mesh = trimesh.Trimesh(vertices = mean_shape_verts, faces = mean_shape_faces)
# mean_shape_mesh.show()

# Extract Expression basis
exp_basis = face_model['exp']     # shape: (107127, 64)
# print("raw exp_flat min/max:", exp_basis.min(), exp_basis.max())
exp_basis = exp_basis.T  # shape: (64, 107127)
exp_basis = exp_basis.reshape(64, -1, 3) # shape: (64, 35709, 3)
# print("exp_basis[0] min/max:", exp_basis[0].min(), exp_basis[0].max())


# Generate Expression Blendshapes

# i = 0

# expr_i = mean_shape_verts + exp_basis[i]

# delta = exp_basis[i]
a = 3.0
# Save expression mesh
for i in range(exp_basis.shape[0]):
    expr_mesh = trimesh.Trimesh(
        vertices = mean_shape_verts + a* exp_basis[i],
        faces    = mean_shape_faces
    )
    expr_mesh.export(f'Output expression mesh/expr_mesh_{i}.ply')