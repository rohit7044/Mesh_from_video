import numpy as np

face_model = np.load('assets/face_model.npy', allow_pickle=True).item()
mean_shape = face_model['u'].squeeze().reshape(-1, 3)

frame_mesh = np.load('Output Video/clipped_video/frame_0000/frame_0000_extractTex.npz')
frame_vertices = frame_mesh['vertices']

vertex_distances = np.linalg.norm(frame_vertices - mean_shape, axis=1)
print("Average vertex-wise distance:", vertex_distances.mean())
print("Max vertex-wise distance:", vertex_distances.max())