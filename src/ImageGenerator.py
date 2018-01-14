import os
from PIL import Image
import random
import threading

BOX_SIZE = 40
IMGS_EACH = 125
IMAGE_SIZE = 500
NUM_SEED_IMGS = 73
NUM_THREADS = 8

def createDict(locationCSV):
    d = {}
    with open(locationCSV) as f:
        for line in f:
            s = line.split(',')
            d[s[0]] = (int(s[1]), int(s[2]))
    return d

def getFiles(imageDirectory):
    l = []
    for filename in os.listdir(imageDirectory):
        l.append(imageDirectory+"/"+filename)
    return l

def generateForFile(img, location, newLocationDictionary, newImageDirectory, randomImages, counter):
    random.seed()
    newFilename = "img%06d.png" % (counter)
    box = (location[0]-BOX_SIZE, location[1]-BOX_SIZE, location[0]+BOX_SIZE, location[1]+BOX_SIZE)
    randomCrop = (location[0]-IMAGE_SIZE//2, location[1]-IMAGE_SIZE//2, location[0]+IMAGE_SIZE//2, location[1]+IMAGE_SIZE//2)
    region = img.crop(box)
    cropRegion = img.crop(randomCrop)
    cropRegion.save(newImageDirectory+"/"+newFilename)
    newLocationDictionary[newFilename] = (IMAGE_SIZE//2 * (IMAGE_SIZE+1), IMAGE_SIZE*IMAGE_SIZE)
    for i in range(0, IMGS_EACH):
        counter += 1
        newFilename = "img%06d.png" % (counter)
        randomFile = randomImages[random.randint(0, len(randomImages)-1)]
        randomImg = Image.open(randomFile)
        s = randomImg.size
        randomLoc = (random.randint(0, s[0]-IMAGE_SIZE), random.randint(0, s[1]-IMAGE_SIZE))
        randomCrop = (randomLoc[0], randomLoc[1], randomLoc[0]+IMAGE_SIZE, randomLoc[1]+IMAGE_SIZE)
        croppedImg = randomImg.crop(randomCrop)
        loc = (random.randint(BOX_SIZE, IMAGE_SIZE-BOX_SIZE), random.randint(BOX_SIZE, IMAGE_SIZE-BOX_SIZE))
        newLocationDictionary[newFilename] = (loc[0]*IMAGE_SIZE+loc[1], IMAGE_SIZE*IMAGE_SIZE)
        newbox = (loc[0]-BOX_SIZE, loc[1]-BOX_SIZE, loc[0]+BOX_SIZE, loc[1]+BOX_SIZE)
        croppedImg.paste(region, newbox)
        croppedImg.save(newImageDirectory+"/"+newFilename)
    return counter

def generate(locationCSV, originalImageDirectory, newImageDirectory, randomImageDirectory, initialCounter):
    locations = createDict(locationCSV)
    randomImages = getFiles(randomImageDirectory)
    newLocationDictionary = {}
    counter = initialCounter
    for filename in os.listdir(originalImageDirectory):
        img = Image.open(originalImageDirectory+"/"+filename)
        counter = generateForFile(img, locations[filename], newLocationDictionary, newImageDirectory, randomImages, counter)
    with open('imagesForTraining.txt', 'a') as f:
        for k, v in newLocationDictionary.items():
            s = k + ", " + str(v[0]) + ", " + str(v[1]) + "\n"
            f.write(s)

if __name__ == '__main__':
    for i in range(NUM_THREADS):
        t = threading.Thread(target=generate, args=("E:\COS 429\FinalProj\stuff.csv", "E:\COS 429\FinalProj\OriginalIms", "E:\COS 429\FinalProj\TheNewIms", "E:\COS 429\FinalProj\RandomIms", i*NUM_SEED_IMGS*IMGS_EACH))
        t.start()