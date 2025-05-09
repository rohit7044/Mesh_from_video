import numpy as np
import trimesh
import os


def load_full_face_model(path):
    model = np.load(path, allow_pickle=True).item()
    mean_shape = model['u'].reshape(-1, 3)
    exp_basis = model['exp'].reshape(-1, 3, model['exp'].shape[1]).transpose(2, 0, 1)
    id_basis = model['id'].reshape(-1, 3, model['id'].shape[1]).transpose(2, 0, 1)
    faces = model['tri']
    return mean_shape, id_basis, exp_basis, faces


def load_blendshape_coefs(path):
    data = np.load(path)
    id_coefs = data['id_coefs']
    exp_coefs = data['exp_coefs']
    return id_coefs, exp_coefs


def reconstruct_mesh(mean_shape, id_basis, exp_basis, id_coefs, exp_coefs):
    id_shape = mean_shape + np.tensordot(id_coefs, id_basis, axes=(0, 0))
    recon_shape = id_shape + np.tensordot(exp_coefs, exp_basis, axes=(0, 0))
    return recon_shape


def main(coef_path, face_model_path, output_path):
    mean_shape, id_basis, exp_basis, faces = load_full_face_model(face_model_path)
    id_coefs, exp_coefs = load_blendshape_coefs(coef_path)

    recon_verts = reconstruct_mesh(mean_shape, id_basis, exp_basis, id_coefs, exp_coefs)
    mesh = trimesh.Trimesh(vertices=recon_verts, faces=faces, process=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.export(output_path)
    print(f"✅ Reconstructed mesh saved to: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Sanity check blendshape coefficients (Identity + Expression)")
    parser.add_argument('--coef_path', required=True, help='Path to .npz file with blendshape coefficients (id_coefs, exp_coefs)')
    parser.add_argument('--face_model', default='assets/face_model.npy', help='Path to face_model.npy')
    parser.add_argument('--output_path', default='blendshape_reconstruction_sanity/reconstructed_mesh.ply', help='Where to save the reconstructed mesh')

    args = parser.parse_args()
    main(args.coef_path, args.face_model, args.output_path)