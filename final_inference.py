import tensorflow as tf
from tensorflow.keras.models import load_model

import keras.backend as K
import cv2
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  


def get_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
            image_paths.append(os.path.join(directory, filename))
    return image_paths



def split_paths(paths, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split a list of paths into train, validation, and test sets.

    Args:
    - paths (list): List of paths to be split.
    - train_ratio (float): Ratio of training set size (default: 0.7).
    - val_ratio (float): Ratio of validation set size (default: 0.15).
    - seed (int): Seed for reproducibility (default: None).

    Returns:
    - train_paths (list): List of paths for the training set.
    - val_paths (list): List of paths for the validation set.
    - test_paths (list): List of paths for the test set.
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)

    # Shuffle the list of paths
    random.shuffle(paths)

    # Calculate sizes for train, validation, and test sets
    total_size = len(paths)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    # Split the shuffled list into train, validation, and test sets
    train_paths = paths[:train_size]
    val_paths = paths[train_size:train_size + val_size]
    test_paths = paths[train_size + val_size:]

    # Ensure test_paths gets the remaining items if sizes don't sum up to total_size
    if len(test_paths) + len(val_paths) + len(train_paths) != total_size:
        test_paths += paths[train_size + val_size + len(test_paths):]

    return train_paths, val_paths, test_paths




def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize images as per your network requirements
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img

# function that compares the test image with a random normal and misprinted image in the trainset

def sub_pred(test_image,model):
    pos = load_image(np.random.choice(train_misprinted_paths))
    neg = load_image(np.random.choice(train_normal_paths))

    anchor_embeddings = model.predict(tf.expand_dims(test_image,axis=0))
    positive_embeddings = model.predict(tf.expand_dims(pos,axis=0))
    negative_embeddings = model.predict(tf.expand_dims(neg,axis=0))
    pos_dist = K.sum(K.square(anchor_embeddings - positive_embeddings), axis=-1).numpy()[0]
    neg_dist = K.sum(K.square(anchor_embeddings - negative_embeddings), axis=-1).numpy()[0]
    if (pos_dist < neg_dist):
        return 1
    else:
        return 0


class Inference:
    def __init__(self, model_path):
        # #transform webcam input to same size to model (normalized)
        #LOAD MODEL

        
        self.model = load_model(model_path)
    

        
    def inference(self, test_image):
        pred1 = sub_pred(test_image,self.model)
        pred2 = sub_pred(test_image,self.model)
        pred3 = sub_pred(test_image,self.model)
        pred4 = sub_pred(test_image,self.model)
        pred5 = sub_pred(test_image,self.model)
        pred6 = sub_pred(test_image,self.model)
        pred7 = sub_pred(test_image,self.model)

        return ((pred1 + pred2 + pred3+pred4 + pred5 + pred6+pred7)//4)

        

        

if __name__ == "__main__":
    # Files link: https://drive.google.com/file/d/1VMottI0IyzCloyLXL7uIRWPTJj7atAFf/view?usp=drive_link
    #replace with paths of your normal and Missing Text folders
    normal_folder = "/kaggle/working/Text 3 images/Text 3 - Training images/Pass"
    misprint_folder = "/kaggle/working/Text 3 images/Text 3 - Training images/Missing Text"

    normal_images_paths = get_image_paths(normal_folder)
    misprinted_images_paths = get_image_paths(misprint_folder)

    # Example usage:
    # Assuming you have a list called normal_images_paths
    train_normal_paths, val_normal_paths, test_normal_paths = split_paths(normal_images_paths)
    train_misprinted_paths, val_misprinted_paths, test_misprinted_paths = split_paths(misprinted_images_paths)


    modelPath = "/kaggle/working/model.h5"
    DLModel = Inference(model_path=modelPath)
    image = load_image("/kaggle/working/Text 3 images/Text 3 - Training images/Missing Text/56.bmp")
    pred_class = DLModel.inference(image) #return predict_class and confidence_score
    if pred_class == 1:
        pred_class = "Misprinted photo"
    else:
        pred_class = "Normal"    
    print("RESULT: {}".format(pred_class))
    
