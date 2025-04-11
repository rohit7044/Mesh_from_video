import argparse
import cv2
import os
import sys
import torch
import numpy as np
from PIL import Image

# Force PyTorch to initialize CUDA first
if torch.cuda.is_available():
    torch.cuda.init()
    # Set default device to CUDA
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from face_box import face_box
from model.recon import face_model
from util.preprocess import get_data_path
from util.io import visualize

def process_frame(frame, recon_model, facebox_detector, args):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame_rgb)
    
    trans_params, im_tensor = facebox_detector(im)
    recon_model.input_img = im_tensor.to(args.device)
    results = recon_model.forward()
    
    return trans_params, results, frame

def main(args):
    recon_model = face_model(args)
    facebox_detector = face_box(args).detector
    
    # Check if input is a video file
    if args.inputpath.lower().endswith(('.mp4', '.avi', '.mov')):
        # Create output directory for video frames
        video_name = os.path.splitext(os.path.basename(args.inputpath))[0]
        output_dir = os.path.join(args.savepath, video_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video file
        cap = cv2.VideoCapture(args.inputpath)
        if not cap.isOpened():
            print(f"Error: Could not open video {args.inputpath}")
            return
            
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            print(f"Processing frame {frame_count}")
            trans_params, results, frame = process_frame(frame, recon_model, facebox_detector, args)
            
            # Visualize and save results
            my_visualize = visualize(results, args)
            frame_output_dir = os.path.join(output_dir, f"frame_{frame_count:04d}")
            os.makedirs(frame_output_dir, exist_ok=True)
            
            my_visualize.visualize_and_output(
                trans_params,
                frame,
                frame_output_dir,
                f"frame_{frame_count:04d}"
            )
            
            frame_count += 1
            
        cap.release()
        print(f"Processed {frame_count} frames")
    else:
        # Process images as before
        im_path = get_data_path(args.inputpath)
        for i in range(len(im_path)):
            print(i, im_path[i])
            im = Image.open(im_path[i]).convert('RGB')
            trans_params, im_tensor = facebox_detector(im)

            recon_model.input_img = im_tensor.to(args.device)
            results = recon_model.forward()

            if not os.path.exists(os.path.join(args.savepath, im_path[i].split('/')[-1].replace('.png','').replace('.jpg',''))):
                os.makedirs(os.path.join(args.savepath, im_path[i].split('/')[-1].replace('.png','').replace('.jpg','')))
            my_visualize = visualize(results, args)

            my_visualize.visualize_and_output(trans_params, cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), \
                os.path.join(args.savepath, im_path[i].split('/')[-1].replace('.png','').replace('.jpg','')), \
                im_path[i].split('/')[-1].replace('.png','').replace('.jpg',''))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA-V3')

    parser.add_argument('-i', '--inputpath', default='examples/', type=str,
                        help='path to the test data, can be an image folder or video file')
    parser.add_argument('-s', '--savepath', default='examples/results', type=str,
                        help='path to the output directory, where results (obj, png files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cuda or cpu' )

    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped and resized into (224,224,3).' )
    parser.add_argument('--detector', default='retinaface', type=str,
                        help='face detector for cropping image, support for mtcnn and retinaface')

    # save
    parser.add_argument('--ldm68', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 68 landmarks')
    parser.add_argument('--ldm106', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 106 landmarks')
    parser.add_argument('--ldm106_2d', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 106 landmarks, face profile is in 2d form')
    parser.add_argument('--ldm134', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 134 landmarks' )
    parser.add_argument('--seg', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show segmentation in 2d without visible mask' )
    parser.add_argument('--seg_visible', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show segmentation in 2d with visible mask' )
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save obj use texture from BFM model')
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save obj use texture extracted from input image')

    # backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='backbone for reconstruction, support for resnet50 and mbnetv3')

    main(parser.parse_args())
