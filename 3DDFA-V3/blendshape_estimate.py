import os
import numpy as np


def load_full_face_model(path):
    model = np.load(path, allow_pickle=True).item()
    mean_shape = model['u'].reshape(-1, 3)
    exp_basis = model['exp'].reshape(-1, 3, model['exp'].shape[1]).transpose(2, 0, 1)
    id_basis = model['id'].reshape(-1, 3, model['id'].shape[1]).transpose(2, 0, 1)
    faces = model['tri']
    return mean_shape, id_basis, exp_basis, faces


def load_mesh_npz(path):
    data = np.load(path)
    V = data['vertices']
    F = data['faces']
    return V.astype(np.float32), F.astype(np.int32)


def estimate_coeffs(mean_shape, basis, target_verts, n_nonzero=20):
    B, V, _ = basis.shape
    basis_flat = basis.reshape(B, V*3)
    residual = (target_verts - mean_shape).reshape(V*3)
    selected_weights = np.zeros(B, dtype=np.float32)
    selected_mask = np.zeros(B, dtype=bool)

    for _ in range(n_nonzero):
        proj_weights = np.array([
            np.dot(basis_flat[i], residual) / (np.dot(basis_flat[i], basis_flat[i]) + 1e-8)
            for i in range(B)
        ])
        proj_weights[selected_mask] = -np.inf
        best_idx = np.argmax(proj_weights)
        best_weight = proj_weights[best_idx]
        if best_weight <= 0:
            break
        selected_weights[best_idx] = best_weight
        selected_mask[best_idx] = True
        residual -= best_weight * basis_flat[best_idx]
    return selected_weights


def main(frames_dir, face_model_path, n_id=20, n_exp=10):
    mean_shape, id_basis, exp_basis, _ = load_full_face_model(face_model_path)
    print("✅ Face model loaded (Identity + Expression basis)")

    out_dir = os.path.join(frames_dir, 'blendshape_coeffs')
    os.makedirs(out_dir, exist_ok=True)

    for frame_folder in sorted(os.listdir(frames_dir)):
        frame_path = os.path.join(frames_dir, frame_folder)
        if not os.path.isdir(frame_path):
            continue

        npz_files = [f for f in os.listdir(frame_path) if f.endswith('.npz')]
        if not npz_files:
            print(f"⚠️ No .npz found in {frame_folder}, skipping")
            continue

        for npz_name in npz_files:
            mesh_file = os.path.join(frame_path, npz_name)
            V, F = load_mesh_npz(mesh_file)

            # Stage 1: Identity fitting
            id_coefs = estimate_coeffs(mean_shape, id_basis, V, n_nonzero=n_id)
            id_shape = mean_shape + np.tensordot(id_coefs, id_basis, axes=(0, 0))

            # Stage 2: Expression fitting on identity-corrected shape
            exp_coefs = estimate_coeffs(id_shape, exp_basis, V, n_nonzero=n_exp)

            # Save coefficients clearly separated
            base = os.path.splitext(npz_name)[0]
            out_name = f"{frame_folder}_{base}_id_exp_coefs.npz"
            out_path = os.path.join(out_dir, out_name)

            np.savez_compressed(out_path, id_coefs=id_coefs, exp_coefs=exp_coefs)
            print(f"✅ {frame_folder}/{npz_name} → saved id/exp coeffs → {out_name}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Two-stage blendshape estimation (Identity + Expression)')
    parser.add_argument('--frames_dir', required=True,
                        help='Directory containing frame_{:04d}/.npz per-frame')
    parser.add_argument('--face_model', default='assets/face_model.npy',
                        help='Path to face_model.npy')
    parser.add_argument('--n_id', type=int, default=20,
                        help='Number of nonzero identity coefficients')
    parser.add_argument('--n_exp', type=int, default=10,
                        help='Number of nonzero expression coefficients')

    args = parser.parse_args()
    main(args.frames_dir, args.face_model, n_id=args.n_id, n_exp=args.n_exp)