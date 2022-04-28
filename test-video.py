import os
import cv2
import time
import torch
import numpy as np
from torchvision.transforms import ToTensor, Compose

from model import Unet
from dataloader import COLOR_CATEGORIES

if __name__ == "__main__":

    # Constant
    IMG_SIZE = (512, 512)
    MODEL_PATH = "./unet_epoch-4.pth"
    VIDEO_PATH = r"F:\ROAD_VIDEO\Clip"

    # Use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Model initialisation
    model = Unet(3, len(COLOR_CATEGORIES))
    model.to(device)

    # Loading the model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Evaluating the model
    model.eval()

    # Image transformation
    img_transform = Compose([ToTensor()])
    
    # Not doing training so no gradient calcul
    with torch.no_grad():

        # Reading videos
        for video_filename in os.listdir(VIDEO_PATH):

            filename = os.path.join(VIDEO_PATH, video_filename)
            cap = cv2.VideoCapture(filename)

            new_frame_time = 0
            prev_frame_time = 0

            # Reading frames
            while(cap.isOpened()):

                ret, frame = cap.read()
                new_frame_time = time.time()

                if not ret:
                    break
                
                # Formatting the frame
                img_resized = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_AREA)
                tensor_img = img_transform(img_resized)
                tensor_img = torch.unsqueeze(tensor_img, dim=0).to(device)

                # Frame -> Infering with the model -> Argmax
                result = model(tensor_img)
                result = torch.squeeze(result, dim=0)
                result_a = torch.argmax(result, dim=0)
                result = result_a.cpu().numpy()

                # Index -> Couleur 
                argmax_result_segmentation = np.expand_dims(result, axis=-1)
                segmentation = np.squeeze(np.take(COLOR_CATEGORIES, argmax_result_segmentation, axis=0))
            
                # Fps calculation
                fps = 1 // (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                # Showing results
                cv2.imshow("ROAD_IMAGE", img_resized)
                cv2.imshow("SEGMENTATION_IMAGE", cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR))

                print(fps)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
