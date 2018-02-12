import argparse
import sys, os
from pathlib import Path
from tqdm import tqdm

import cv2

CASCADE_PATH = "data/haarcascade_frontalface_default.xml"

def detect_face(image, minsize):
    """
    Detect face region in image

    :params np.ndarray image: input image
    :params int minsize: face rectangle size
    """

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    facerect = cascade.detectMultiScale(\
                image_gray, scaleFactor=1.1, \
                minNeighbors=2, minSize=(minsize, minsize)
               )
    rect = facerect[0] if len(facerect) > 0 else None

    return rect

def preprocess_celeb_a(root_path, save_path):
    """
    Perfom preprocessing for CelebA dataset
    
    :param str root_path: root path of dataset
    :param str save_path: save path of preprocessed dataset
    """
    image_paths = list(root_path.glob("*.jpg"))
    size = 128

    for input_path in tqdm(image_paths):
        img = cv2.imread(str(input_path))
        box = detect_face(img, size)
        
        out_path = save_path / input_path.name
        if box is not None and box[2] > size and box[3] > size:
            face_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            cv2.imwrite(str(out_path), face_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_type', choices=['celeb_a'], default="celeb_a")
    parser.add_argument('root_path')
    parser.add_argument('save_path', default='train_iamges/celeb_a')
    args = parser.parse_args()
    
    root_path = Path(args.root_path)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    preprocess_funcs = {
        'celeb_a': preprocess_celeb_a, 
    }
    
    preprocess_funcs[args.dataset_type](root_path, save_path)

if __name__=="__main__":
    main()
