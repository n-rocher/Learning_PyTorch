from dataloader import A2D2_Dataset
import cv2
import numpy as np
from tqdm import tqdm
import random

if __name__ == "__main__":

    training_dataset = A2D2_Dataset("training", size=(512, 400))

    total = np.zeros((400, 512, 3))

    range_ = (list(range(len(training_dataset))))
    random.shuffle(range_)

    for id in tqdm(range_[:2000], total=2000):
        img, target = training_dataset.__getitem__(id)
        total += img
        print()
        print("", np.max(total))

        # cv2.imshow("img", img.numpy().transpose(1, 2, 0))
        # cv2.imshow("target", np.argmax(target.numpy(), axis=0)/16)

        # cv2.waitKey(0)
    np.save("total_A2D2.numpy", total)
