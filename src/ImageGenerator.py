import os
from PIL import Image
import random

BOX_SIZE = 40

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
    newFilename = "img%06d.jpg" % (counter)
    newLocationDictionary[newFilename] = (location[0], location[1])
    box = (location[0]-BOX_SIZE, location[1]-BOX_SIZE, location[0]+BOX_SIZE, location[1]+BOX_SIZE)
    region = img.crop(box)
    img.save(newImageDirectory+"/"+newFilename)
    for i in range(0, 1000):
        counter += 1
        newFilename = "img%06.jpg" % (counter)
        randomFile = randomImages[random.randint(0, len(randomImages)-1)]
        randomImg = Image.open(randomFile)
        s = randomImg.size
        loc = (random.randint(BOX_SIZE, s[0]-BOX_SIZE), random.randint(BOX_SIZE, s[1]-BOX_SIZE))
        newLocationDictionary[newFilename] = loc
        newbox = (loc[0]-BOX_SIZE, loc[1]-BOX_SIZE, loc[0]+BOX_SIZE, loc[1]+BOX_SIZE)
        randomImg.paste(region, newbox)
        randomImg.save(newImageDirectory+"/"+newFilename)
    return counter

def generate(locationCSV, originalImageDirectory, newImageDirectory, randomImageDirectory):
    locations = createDict(locationCSV)
    randomImages = getFiles(randomImageDirectory)
    newLocationDictionary = {}
    counter = 0
    for filename in os.listdir(originalImageDirectory):
        img = Image.open(originalImageDirectory+"/"+filename)
        counter = generateForFile(img, locations[filename], newLocationDictionary, newImageDirectory, randomImages, counter)
    with open('trainingImages.txt', 'w') as f:
        for k, v in newLocationDictionary:
            s = k + ", " + str(v[0]) + ", " + str(v[1]) + "\n"
            f.write(s)
