import cv2
import os
import torch
import numpy as np
import time
from PIL import Image

from face_box import face_box
from model.recon import face_model
from util.io import visualize, back_resize_crop_img, plot_kpts, back_resize_ldms


def main():
    # Define default args without using argparse
    class Args:
        def __init__(self):
            self.inputpath = 'examples/'
            self.savepath = 'examples/results'
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.iscrop = True
            self.detector = 'retinaface'
            self.ldm68 = True
            self.ldm106 = True
            self.ldm106_2d = True
            self.ldm134 = True
            self.seg = True
            self.seg_visible = True
            self.useTex = True
            self.extractTex = True
            self.backbone = 'resnet50'

    args = Args()

    # Create directory for saving results if needed
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    # Initialize models
    recon_model = face_model(args)
    facebox_detector = face_box(args).detector

    # Start webcam
    cap = cv2.VideoCapture(0)

    print("Webcam started. Press 's' to save a frame, 'q' to quit.")

    while cap.isOpened():
        ret, cv2_im = cap.read()

        if not ret:
            print("Failed to get frame from webcam")
            break

        # Convert to PIL image for face detection
        im = Image.fromarray(cv2_im).convert('RGB')
        trans_params, im_tensor = facebox_detector(im)

        # Only proceed if face is detected
        if im_tensor is not None:
            # Process image through 3D face model
            recon_model.input_img = im_tensor.to(args.device)
            results = recon_model.forward()

            # Create a copy for visualization
            visualization_img = cv2_im.copy()
            img_bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

            # Display 3D shape overlay
            render_shape = (results['render_shape'][0] * 255).astype(np.uint8)
            render_mask = (np.stack((results['render_mask'][0][:, :, 0],) * 3, axis=-1) * 255).astype(np.uint8)

            if trans_params is not None:
                render_shape = back_resize_crop_img(render_shape, trans_params, np.zeros_like(img_bgr),
                                                    resample_method=Image.BICUBIC)
                render_mask = back_resize_crop_img(render_mask, trans_params, np.zeros_like(img_bgr),
                                                   resample_method=Image.NEAREST)

            # Blend overlay with original image
            visualization_img = ((render_shape / 255. * render_mask / 255. + img_bgr[:, :, ::-1] / 255. * (
                        1 - render_mask / 255.)) * 255).astype(np.uint8)[:, :, ::-1]

            # Display landmarks if enabled
            if args.ldm68 and 'ldm68' in results:
                ldm68 = results['ldm68'][0].copy()
                ldm68[:, 1] = 224 - 1 - ldm68[:, 1]
                if trans_params is not None:
                    ldm68 = back_resize_ldms(ldm68, trans_params)
                visualization_img = plot_kpts(visualization_img, ldm68)

            # Show the result
            cv2.imshow('3DDFA-V3 Live Overlay', visualization_img)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Save the current frame with full visualization
                frame_name = f"frame_{int(time.time())}"
                my_visualize = visualize(results, args)
                my_visualize.visualize_and_output(trans_params, img_bgr, args.savepath, frame_name)
                print(f"Saved frame as {frame_name}")
            elif key == ord('q'):
                break
        else:
            # Show original frame if no face detected
            cv2.putText(cv2_im, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('3DDFA-V3 Live Overlay', cv2_im)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()