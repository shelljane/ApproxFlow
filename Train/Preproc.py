import cv2
import random
import numpy as np

def indicesInverted(data, labelsCoarse, labelsFine): 
    assert isinstance(data, np.ndarray), 'data must be numpy.ndarray'
    assert data.shape[0] == len(labelsCoarse) == len(labelsFine), 'data and labels must match in size'
    indicesCoarse = [[] for _ in range(np.max(labelsCoarse)+1)]
    indicesFine   = [[] for _ in range(np.max(labelsFine)+1)]
    for idx in range(data.shape[0]):
        labelCoarse = labelsCoarse[idx]
        labelFine   = labelsFine[idx]
        indicesCoarse[labelCoarse].append(idx)
        indicesFine[labelFine].append(idx)
    
    return indicesCoarse, indicesFine

def genIndex(size, shuffle=True): 
    perm = list(range(size))
    if shuffle:
        random.shuffle(perm)
    index = 0
    while True:
        if index >= size: 
            index = 0
            perm = list(range(size))
            if shuffle:
                random.shuffle(perm)
        yield perm[index]
        index += 1
        
def randomCrop(image, size): 
    beginX = random.randint(0, image.shape[0]-size[0])
    beginY = random.randint(0, image.shape[1]-size[1])     
    
    return image[beginX:beginX+size[0], beginY:beginY+size[1], :]
        
def centerCrop(image, size): 
    beginX = int((image.shape[0]-size[0]) / 2)
    beginY = int((image.shape[1]-size[1]) / 2)
    
    return image[beginX:beginX+size[0], beginY:beginY+size[1], :]
    
def randomRotate(image, rng=10): 
    (h, w) = image.shape[0:2]
    angle = (random.random() - 0.5) * rng * 2
    mat = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

    return cv2.warpAffine(image, mat, (w, h))

def randomFlip(image): 
    flipped = image
    if random.random() > 0.5: 
        flipped = cv2.flip(flipped, 1) # Flipped Horizontally

    if random.random() > 0.5: 
        flipped = cv2.flip(flipped, 0) # Flipped Vertically
    
    return flipped

def randomFlipH(image): 
    flipped = image
    if random.random() > 0.5: 
        flipped = cv2.flip(flipped, 1) # Flipped Horizontally
    
    return flipped
    
def randomBrightness(image, rng=10):
    b = int((random.random()-0.5) * rng * 2)
    return np.uint8(np.clip(np.int32(image) + b, 0, 255))
    
def randomContrast(image, mini=0.5, maxi=1.5):
    a = mini + (maxi-mini) * random.random()
    b = 125 * (1 - a)
    return np.uint8(np.clip(a * image + b, 0, 255))

def randomShift(image, rng=4): 
    shiftX = random.randint(-rng, rng)
    shiftY = random.randint(-rng, rng)
    return np.roll(np.roll(image, shiftX, axis=0), shiftY, axis=1)
    
    
    
