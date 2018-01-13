import os
from PIL import Image
import random
import threading

BOX_SIZE = 40
IMGS_EACH = 50
MIN_IMAGE_SIZE = 350
MAX_IMAGE_SIZE = 450
NUM_SEED_IMGS = 3
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
    newFilename = "img%06d.jpg" % (counter)
    box = (location[0]-BOX_SIZE, location[1]-BOX_SIZE, location[0]+BOX_SIZE, location[1]+BOX_SIZE)
    randomX, randomY = random.randint(MIN_IMAGE_SIZE, MAX_IMAGE_SIZE), random.randint(MIN_IMAGE_SIZE, MAX_IMAGE_SIZE)
    randomCrop = (location[0]-(randomX//2), location[1]-(randomY//2), location[0]+(randomX//2), location[1]+randomY//2)
    region = img.crop(box)
    cropRegion = img.crop(randomCrop)
    cropRegion.save(newImageDirectory+"/"+newFilename)
    newLocationDictionary[newFilename] = ((randomX//2)*randomY+(randomY)//2, randomX*randomY)
    for i in range(0, IMGS_EACH):
        counter += 1
        newFilename = "img%06d.jpg" % (counter)
        randomFile = randomImages[random.randint(0, len(randomImages)-1)]
        randomImg = Image.open(randomFile)
        s = randomImg.size
        randomX, randomY = random.randint(MIN_IMAGE_SIZE, MAX_IMAGE_SIZE), random.randint(MIN_IMAGE_SIZE,MAX_IMAGE_SIZE)
        randomLoc = (random.randint(0, s[0]-randomX), random.randint(0, s[1]-randomY))
        randomCrop = (randomLoc[0], randomLoc[1], randomLoc[0]+randomX, randomLoc[1]+randomY)
        croppedImg = randomImg.crop(randomCrop)
        loc = (random.randint(BOX_SIZE, randomX-BOX_SIZE), random.randint(BOX_SIZE, randomY-BOX_SIZE))
        newLocationDictionary[newFilename] = (loc[0]*randomY+loc[1], randomX*randomY)
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
    with open('trainingImages.txt', 'w') as f:
        for k, v in newLocationDictionary.items():
            s = k + ", " + str(v[0]) + ", " + str(v[1]) + "\n"
            f.write(s)

if __name__ == '__main__':
    for i in range(NUM_THREADS):
        t = threading.Thread(target=generate, args=("C:\WaldoIms\stuff.csv", "C:\WaldoIms\WaldoDir", "C:\WaldoIms\TheNewDir", "C:\WaldoIms\RandomImDir", i*NUM_SEED_IMGS*IMGS_EACH))
        t.start()