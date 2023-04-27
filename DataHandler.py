import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from torch.utils.data import Dataset, DataLoader
import torch

dataset_locs = {
    "X_HH" : "C:\\Users\\tomms\\Desktop\\DL Exam\\healthcare_data\\subcortical\\X_HH.npy",
    "y_HH" : "C:\\Users\\tomms\\Desktop\\DL Exam\\healthcare_data\\subcortical\\y_HH.npy",
    "ids_HH" : "C:\\Users\\tomms\\Desktop\\DL Exam\\healthcare_data\\subcortical\\ids_HH.npy",
    "X_Guys" : "C:\\Users\\tomms\\Desktop\\DL Exam\\healthcare_data\\subcortical\\X_Guys.npy",
    "y_Guys" : "C:\\Users\\tomms\\Desktop\\DL Exam\\healthcare_data\\subcortical\\y_Guys.npy",
    "ids_Guys" : "C:\\Users\\tomms\\Desktop\\DL Exam\\healthcare_data\\subcortical\\ids_Guys.npy",
    "X_IOP" : "C:\\Users\\tomms\\Desktop\\DL Exam\\healthcare_data\\subcortical\\X_IOP.npy",
    "y_IOP" :"C:\\Users\\tomms\\Desktop\\DL Exam\\healthcare_data\\subcortical\\y_IOP.npy",
    "y_ids" : "C:\\Users\\tomms\\Desktop\\DL Exam\\healthcare_data\\subcortical\\y_ids.npy"
}

dataset_names = { 
    "HH" : ["X_HH", "y_HH"], 
    "Guys" : ["X_Guys", "y_Guys"],
    "IOP" : ["X_IOP", "y_IOP"]
}

def standardise(X):
    means = []
    stdDevs = [] 
    eps = 1e-7 #prevent div by 0 
    for x in X:
        means.append(np.mean(x, keepdims = True))
        stdDevs.append(np.std(x, keepdims = True))
    mean = np.mean(means)
    stdDev = np.mean(stdDevs) + eps
    print("mean: ", mean) 
    print("stdDev: ", stdDev)
    return (X-mean)/stdDev 
    
def normalise(X):
    mins = [] 
    maxs = [] 
    for x in X:
        mins.append(np.min(x, keepdims = True)) 
        maxs.append(np.max(x, keepdims = True)) 
    min = np.min(mins)
    max = np.max(maxs)
    return (X-min)/max

def to_one_hot(y, class_loc, numClasses = 5): #class_loc is 0 for dice_score and 1 for data prep 
    store = to_categorical(y, num_classes = numClasses) #puts the classes at the end so we need to move them 
    return np.moveaxis(store, store.ndim-1, class_loc)

class numpy_dataset(Dataset):
    def __init__(self, data, target):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.from_numpy(data).float() #convert the numpy arrays into pytorch tensors 
        # self.data = self.data.to(device) #put the tensors on the gpu if possible 
        self.target = target #Target is just the labels so it doesn't need converting here I think we one_hot it elsewhere, we could've done it here though 

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data) 

def mergeFirstTwoDims(X): 
    return X.reshape(X.shape[0]*X.shape[1],X.shape[2],X.shape[3])

def getSplitData(datasets, dimensions = 3, training_size = 0.8, test_val_size = 0.5):
    xs = []
    ys = []
    #concatenate all the datasets together
    for dataset in datasets:
        print("loading ", dataset)
        xs.append(np.load(dataset_locs["X_" + dataset]))
        ys.append(np.load(dataset_locs["y_" + dataset]))
    print("concatenating datasets")
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    print("splitting data")
    X = standardise(X)
    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, train_size=training_size, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size = test_val_size, shuffle=True)

    #if we want it in 2D, once we've split into the sets, we slice
    if dimensions == 2: 
        X_train = mergeFirstTwoDims(X_train)
        X_val = mergeFirstTwoDims(X_val)
        X_test = mergeFirstTwoDims(X_test)
        y_train = mergeFirstTwoDims(y_train)
        y_val = mergeFirstTwoDims(y_val)
        y_test = mergeFirstTwoDims(y_test)

    print("adding new axis to X")
    X_train = X_train[:,np.newaxis,:,:]
    X_val = X_val[:,np.newaxis,:,:]
    X_test = X_test[:,np.newaxis,:,:]

    print("one-hotting y")
    y_train = to_one_hot(y_train, class_loc = 1)
    y_val = to_one_hot(y_val, class_loc = 1)
    y_test = to_one_hot(y_test, class_loc = 1)

    return [X_train, X_val, X_test, y_train, y_val, y_test]

def getTestingData(datasets, dimensions = 3):
    xs = []
    ys = []
    #concatenate all the datasets together
    for dataset in datasets:
        print("loading ", dataset)
        xs.append(np.load(dataset_locs["X_" + dataset]))
        ys.append(np.load(dataset_locs["y_" + dataset]))
    print("concatenating datasets")
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    print("splitting data")
    X = standardise(X)

    if dimensions == 2: 
        X = mergeFirstTwoDims(X)
        y = mergeFirstTwoDims(y)
        
    print("adding new axis to X")
    X = X[:, np.newaxis,:,:]

    print("one-hotting y")
    y = to_one_hot(y, class_loc = 1)

    return [X, y]

