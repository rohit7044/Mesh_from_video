import trimesh
import numpy as np
import os

from collections import deque

# Face nesh Path
# face_mesh_path = "Output Video"
# face_model_path = "assets/face_model.npy"

# Read mesh

def read_meshes(path):
    meshes = []
    # Get all frame folders (they are named frame_XXXX)
    frame_folders = [f for f in os.listdir(path) if f.startswith('frame_') and os.path.isdir(os.path.join(path, f))]
    # Sort frame folders to ensure correct order
    frame_folders.sort()
    
    for frame_folder in frame_folders:
        frame_path = os.path.join(path, frame_folder)
        # Look for .obj files with '_extractTex' in their names
        for file in os.listdir(frame_path):
            if file.endswith(".obj") and "_extractTex" in file:
                mesh_path = os.path.join(frame_path, file)
                meshes.append(trimesh.load(mesh_path))
                break  # Found the extractTex mesh for this frame
    
    return meshes

def load_face_model(path):
    return np.load(path,allow_pickle=True).item()

def geodesic_expand(faces, seed_idxs, hops=2):
    """
    Given a mesh’s faces and a set of seed vertex‐indices,
    returns all vertices within `hops` edge‐steps of any seed.
    """
    V = faces.max() + 1
    # build adjacency list
    adj = [[] for _ in range(V)]
    for f in faces:
        a,b,c = f
        adj[a].extend([b,c])
        adj[b].extend([a,c])
        adj[c].extend([a,b])

    visited = set(seed_idxs)
    queue = deque((v,0) for v in seed_idxs)
    while queue:
        v, d = queue.popleft()
        if d >= hops: 
            continue
        for nbr in adj[v]:
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, d+1))
    return np.array(sorted(visited), dtype=int)

def mask_region(mesh, face_model, part_indices, hops=10):
    """
    Extracts a sub‑mesh around the given `part_indices` expanded
    by `hops` geodesic steps.
    - mesh: mesh object
    - face_model: the dict from face_model.npy
    - part_indices: list of mesh-vertex IDs to start from (e.g. lips)
    - hops: number of rings to expand (2–3 is typical)
    """
    V = mesh.vertices
    F = mesh.faces

    # 1) expand the region
    region_idxs = geodesic_expand(F, part_indices, hops=hops)

    # 2) slice vertices
    V_reg = V[region_idxs]

    # 3) pick only faces whose all 3 verts are in region
    is_in = np.zeros(len(V), dtype=bool)
    is_in[region_idxs] = True
    F_reg_orig = [face for face in F
                  if is_in[face[0]] and is_in[face[1]] and is_in[face[2]]]
    F_reg_orig = np.array(F_reg_orig, dtype=int)

    # 4) remap original vertex IDs -> new 0…R-1 indexing
    remap = {orig: new for new, orig in enumerate(region_idxs)}
    F_reg = np.array([[remap[v] for v in face] for face in F_reg_orig],
                     dtype=int)

    # 5) build sub‑mesh
    return trimesh.Trimesh(vertices=V_reg, faces=F_reg, process=False)

# if __name__ == "__main__":
#     # meshes = read_meshes(face_mesh_path)
#     face_model = load_face_model(face_model_path)
#     annotations = face_model['annotation']
#     # get your lip seeds (upper + lower)
#     lip_seeds = np.concatenate([annotations[5], annotations[6]]).astype(int)

#     # extract a lip + 2‑hop surrounding patch
#     patch = mask_region("Output Video/frame_0001/frame_0001_extractTex.obj", face_model, lip_seeds, hops=10)
#     patch.export("Output expression mesh/lip_mesh.ply")
