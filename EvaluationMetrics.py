import numpy as np
from DataHandler import to_one_hot
def dice_single_class(pred, true): 
    for i in range(len(pred)): 
        pred_f = pred.flatten() 
        true_f = true.flatten()
        eps = 1e-7
        intersection = np.sum(pred_f*true_f) 
        # print("intersection: ", intersection) 
        cardinality = np.sum(pred_f + true_f)
        # print("Cardinality: ", cardinality)
        return (2*intersection)/(cardinality + eps)
labels = ["Blank space","Thalamus","Caudate","Putamen","Amygdala"]

def dice_score(pred, true, verbose = True, num_classes=5): 
    pred_oh = to_one_hot(pred, class_loc = 0, numClasses = num_classes)
    true_oh = to_one_hot(true, class_loc = 0, numClasses = num_classes) 
    # print("pred_oh shape: ", pred_oh.shape)
    scores = [] #the scores for each class 
    for c in range(0,num_classes):
        scores.append(dice_single_class(pred_oh[c], true_oh[c]))
        if verbose:
            print("class ", labels[c], " score: ", scores[c])
    return scores