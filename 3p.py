from PIL import Image
from multiprocessing import Process, Manager
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from os import walk
from math import sqrt
from collections import deque
import pickle
from scipy.misc import imsave
import sys
from copy import deepcopy

def readGreyScale(name=None, f=lambda x: x):
    if name == None:
        return None
    im = Image.open(name).convert('L')
    m, n = im.size

    red_v = []

    for i in range(m):

        red_v.append([])

        for j in range(n):
            red_v[i].append(f(im.getpixel((i, j))))

    return red_v

#Reads image and puts it in the right format
def readImage(name=None):
    if name == None:
        return None
    im = Image.open(name).convert('RGB')
    m, n = im.size

    red_v = []

    green_v = []

    blue_v = []

    for i in range(m):

        red_v.append([])

        green_v.append([])

        blue_v.append([])

        for j in range(n):
            r, g, b = im.getpixel((i, j))

            red_v[i].append(r)

            green_v[i].append(g)

            blue_v[i].append(b)
    return (red_v, green_v, blue_v)

def save_as_img(ar, fname):
    Image.fromarray(ar.round().astype(np.uint8)).save(fname)

def save_rgb_img(red_v, green_v, blue_v, fname):
    imsave(fname, np.dstack((red_v, green_v, blue_v)))

#Processes an example output image
def getSingleCellResult(image):
    #global test
    red_v, green_v, blue_v = image
    rez = []
    n = len(red_v)
    m = len(red_v[0])

    for i in range(n):

        rez.append([])

        j = 0

        while j < m and red_v[i][j] + green_v[i][j] + blue_v[i][j] != 0:
            rez[i].append(0)
            j = j + 1

        while j < m:
            if red_v[i][j] + green_v[i][j] + blue_v[i][j] == 0:
                rez[i].append(1)
            else:
                foundBound = False

                #sus
                for ii in range(i - 1, -1, -1):
                    if red_v[ii][j] + green_v[ii][j] + blue_v[ii][j] == 0:
                        foundBound = True
                        break

                if foundBound:
                    foundBound = False
                    #jos
                    for ii in range(i + 1, n):
                        if red_v[ii][j] + green_v[ii][j] + blue_v[ii][j] == 0:
                            foundBound = True
                            break

                    if foundBound:
                        foundBound = False
                        #stanga
                        for jj in range(j-1, -1, -1):
                            if red_v[i][jj] + green_v[i][jj] + blue_v[i][jj] == 0:
                                foundBound = True
                                break

                        if foundBound:
                            foundBound = False
                            #dreapta
                            for jj in range(j+1, m):
                                if red_v[i][jj] + green_v[i][jj] + blue_v[i][jj] == 0:
                                    foundBound = True
                                    break

                            if foundBound:
                                rez[i].append(1)
                if not foundBound:
                    rez[i].append(0)

            j = j + 1
    return rez

#Builds feature arrays
def buildFeatureArray(red_v, green_v, blue_v, radius, radius1, radius2):
    inputs = []

    n = len(red_v)
    m = len(red_v[0])
    val = round(sqrt(radius * radius / 2))
    val1 = round(sqrt(radius1 * radius1 / 2))

    for i in range(n):
        for j in range(m):
            features = [\
                \
                red_v[i][j],\
                green_v[i][j],\
                blue_v[i][j],\
                \
                red_v[i][j-radius] if j-radius >= 0 else 0,\
                green_v[i][j-radius] if j-radius >= 0 else 0,\
                blue_v[i][j-radius] if j-radius >= 0 else 0,\
                \
                red_v[i][j+radius] if j+radius < m else 0,\
                green_v[i][j+radius] if j+radius < m else 0,\
                blue_v[i][j+radius] if j+radius < m else 0,\
                \
                red_v[i-radius][j] if i-radius >= 0 else 0,\
                green_v[i-radius][j] if i-radius >= 0 else 0,\
                blue_v[i-radius][j] if i-radius >= 0 else 0,\
                \
                red_v[i+radius][j] if i+radius < n else 0,\
                green_v[i+radius][j] if i+radius < n else 0,\
                blue_v[i+radius][j] if i+radius < n else 0,\
                \
                red_v[i+val][j+val] if i+val < n and j+val < m else 0,\
                green_v[i+val][j+val] if i+val < n and j+val < m else 0,\
                blue_v[i+val][j+val] if i+val < n and j+val < m else 0,\
                \
                red_v[i-val][j+val] if i-val >= 0 and j+val < m else 0,\
                green_v[i-val][j+val] if i-val >= 0 and j+val < m else 0,\
                blue_v[i-val][j+val] if i-val >= 0 and j+val < m else 0,\
                \
                red_v[i+val][j-val] if i+val < n and j-val >= 0 else 0,\
                green_v[i+val][j-val] if i+val < n and j-val >= 0 else 0,\
                blue_v[i+val][j-val] if i+val < n and j-val >= 0 else 0,\
                \
                red_v[i-val][j-val] if i-val >= 0 and j-val >= 0 else 0,\
                green_v[i-val][j-val] if i-val >= 0 and j-val >= 0 else 0,\
                blue_v[i-val][j-val] if i-val >= 0 and j-val >= 0 else 0,\
                \
                red_v[i][j-radius1] if j-radius1 >= 0 else 0,\
                green_v[i][j-radius1] if j-radius1 >= 0 else 0,\
                blue_v[i][j-radius1] if j-radius1 >= 0 else 0,\
                \
                red_v[i][j+radius1] if j+radius1 < m else 0,\
                green_v[i][j+radius1] if j+radius1 < m else 0,\
                blue_v[i][j+radius1] if j+radius1 < m else 0,\
                \
                red_v[i-radius1][j] if i-radius1 >= 0 else 0,\
                green_v[i-radius1][j] if i-radius1 >= 0 else 0,\
                blue_v[i-radius1][j] if i-radius1 >= 0 else 0,\
                \
                red_v[i+radius1][j] if i+radius1 < n else 0,\
                green_v[i+radius1][j] if i+radius1 < n else 0,\
                blue_v[i+radius1][j] if i+radius1 < n else 0,\
                \
                red_v[i+val1][j+val1] if i+val1 < n and j+val1 < m else 0,\
                green_v[i+val1][j+val1] if i+val1 < n and j+val1 < m else 0,\
                blue_v[i+val1][j+val1] if i+val1 < n and j+val1 < m else 0,\
                \
                red_v[i-val1][j+val1] if i-val1 >= 0 and j+val1 < m else 0,\
                green_v[i-val1][j+val1] if i-val1 >= 0 and j+val1 < m else 0,\
                blue_v[i-val1][j+val1] if i-val1 >= 0 and j+val1 < m else 0,\
                \
                red_v[i+val1][j-val1] if i+val1 < n and j-val1 >= 0 else 0,\
                green_v[i+val1][j-val1] if i+val1 < n and j-val1 >= 0 else 0,\
                blue_v[i+val1][j-val1] if i+val1 < n and j-val1 >= 0 else 0,\
                \
                red_v[i-val1][j-val1] if i-val1 >= 0 and j-val1 >= 0 else 0,\
                green_v[i-val1][j-val1] if i-val1 >= 0 and j-val1 >= 0 else 0,\
                blue_v[i-val1][j-val1] if i-val1 >= 0 and j-val1 >= 0 else 0,\
                \
                red_v[i][j-radius2] if j-radius2 >= 0 else 0,\
                green_v[i][j-radius2] if j-radius2 >= 0 else 0,\
                blue_v[i][j-radius2] if j-radius2 >= 0 else 0,\
                \
                red_v[i][j+radius2] if j+radius2 < m else 0,\
                green_v[i][j+radius2] if j+radius2 < m else 0,\
                blue_v[i][j+radius2] if j+radius2 < m else 0,\
                \
                red_v[i-radius2][j] if i-radius2 >= 0 else 0,\
                green_v[i-radius2][j] if i-radius2 >= 0 else 0,\
                blue_v[i-radius2][j] if i-radius2 >= 0 else 0,\
                \
                red_v[i+radius2][j] if i+radius2 < n else 0,\
                green_v[i+radius2][j] if i+radius2 < n else 0,\
                blue_v[i+radius2][j] if i+radius2 < n else 0\
                ]
            inputs.append(features)
    return inputs

#Erases white spots
def filterLittleWhiteSpots(rez):
    n = len(rez)
    m = len(rez[0])

    coordModifiers = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    whiteSpots = []
    maxAreaWhiteSpot = None
    for i in range(n):
        for j in range(m):
            if rez[i][j] == 1:

                flag = False
                for whiteSpot in whiteSpots:
                    if (i,j) in whiteSpot[0]:
                        flag = True
                        break

                if not flag:#Found new white spot !
                    newWhiteSpotCoords = [(i,j)]
                    areaValue = 1
                    queue = deque([(i, j)])
                    while len(queue) != 0:
                        coords = queue.popleft()

                        for coordModifier in coordModifiers:
                            newI = coords[0] + coordModifier[0]
                            if 0 <= newI and newI < n:
                                newJ = coords[1] + coordModifier[1]
                                if 0 <= newJ and newJ < m\
                                    and rez[newI][newJ] == 1\
                                    and (newI, newJ) not in newWhiteSpotCoords:
                                    queue.append((newI, newJ))
                                    newWhiteSpotCoords.append((newI, newJ))
                                    areaValue = areaValue + 1

                    if maxAreaWhiteSpot == None or maxAreaWhiteSpot[1] < areaValue:
                        maxAreaWhiteSpot = (newWhiteSpotCoords, areaValue)

                    whiteSpots.append((newWhiteSpotCoords, areaValue))

    if len(whiteSpots) == 1:
        return rez

    whiteSpots.pop(whiteSpots.index(maxAreaWhiteSpot))

    for whiteSpot in whiteSpots:
        for coords in whiteSpot[0]:
            rez[coords[0]][coords[1]] = 0

    return rez

def arrayToMatrix(vec, n, m, f=lambda x: x):
    imgRez = []
    k = 0
    for i in range(n):
        imgRez.append([])
        for j in range(m):
            imgRez[i].append(f(vec[k]))
            k = k + 1
    return imgRez

# predicted is a one dimensional vector
def writeToFile(name, predicted, n, m):
    save_as_img(\
        np.asarray(\
            arrayToMatrix(\
                predicted,\
                n,\
                m,\
                lambda x: 255 if x == 1 else 0\
    )).transpose(), './R/' + name)
    #save_as_img(np.asarray(filterLittleWhiteSpots(imgRez)).transpose(), './R/' + name)        n = len(red_v)

# Returns a part of an image for "im" parameter
def trimImage(im, center, width, height, f=lambda x: x):
    rez = []
    n = len(im)
    m = len(im[0])
    ind = 0

    for i in range(int(center[0] - (width-1)/2), int(center[0] + (width-1)/2 + 1)):
        rez.append([])
        for j in range(int(center[1] - (height-1)/2), int(center[1] + (height-1)/2 + 1)):
            rez[ind].append(f(im[i][j]) if 0 <= i and i < n and 0 <= j and j < m else f(0))
        ind = ind + 1

    return rez

def getDimensions(refCenter, newCenter, refWidth, refHeight, maxWidth, maxHeight):
    difI = int(abs(refCenter[0] - newCenter[0]) + refWidth)
    difJ = int(abs(refCenter[1] - newCenter[1]) + refHeight)

    if difI % 2 == 1:
        difI = difI + 1

    if difJ % 2 == 1:
        difJ = difJ + 1

    return (\
        maxWidth if maxWidth >= difI else difI,\
        maxHeight if maxHeight >= difJ else difJ)

if __name__ == "__main__":

    radius = 7
    radius1 = 28
    radius2 = 112

    engine = pickle.load(open('finalized_model.sav', 'rb'))

    n = -1
    data = []
    for name in filter(lambda n: n.endswith(".tif"), list(walk(sys.argv[1]))[0][2]):
        print("computin: " + str(name))

        red_v, green_v, blue_v = readImage(sys.argv[1] + name)

        if n == -1:
            n = len(red_v)
            m = len(red_v[0])

        data.append((\
            red_v,\
            green_v,\
            blue_v,\
            filterLittleWhiteSpots(\
                arrayToMatrix(\
                    engine.predict(\
                        buildFeatureArray(red_v, green_v, blue_v, radius, radius1, radius2)\
                    ),\
                    n,\
                    m\
                )\
            ),\
            name,\
        ))

    GEOM_CENTER = "geomCenter"
    MASS_CENTER = "massCenter"
    RED_CENTER = "redCenter"
    POND_CENTER = "pondCenter"
    centers = []
    dimensionsMaxes = {\
        GEOM_CENTER : (-1, -1),\
        RED_CENTER : (-1, -1),\
        MASS_CENTER : (-1, -1),\
        POND_CENTER : (-1, -1)\
    }
    for picture in data:
        minI = 5000
        minJ = minI
        maxI = -1
        maxJ = maxI
        redSumI = 0
        redSumJ = 0
        redCount = 0
        centOfMassI = 0
        centOfMassJ = 0
        centOfMassCount = 0
        pondSumI = 0
        pondSumJ = 0
        pondSum = 0

        n = len(picture[0])
        m = len(picture[0][0])

        f = open(sys.argv[1] + picture[4][:-2] + "xt", "rt")
        content = list(map(lambda line: line.split(), f.readlines()))[4:]
        f.close()

        for i in range(n):
            for j in range(m):
                if picture[3][i][j] == 1:
                    if minI > i:
                        minI = i
                    if maxI < i:
                        maxI = i
                    if minJ > j:
                        minJ = j
                    if maxJ < j:
                        maxJ = j

                    centOfMassI = centOfMassI + i
                    centOfMassJ = centOfMassJ + j
                    centOfMassCount = centOfMassCount + 1

                    pondSumI = pondSumI + i * float(content[j][i])
                    pondSumJ = pondSumJ + j * float(content[j][i])
                    pondSum = pondSum + float(content[j][i])
                if picture[0][i][j] == 127 and picture[1][i][j] == 0 and picture[2][i][j] == 0:
                    redSumI = redSumI + i
                    redSumJ = redSumJ + j
                    redCount = redCount + 1

        if (maxI - minI) % 2 != 0:
            if minI > 0:
                minI = minI - 1
            elif maxI < n-1:
                maxI = maxI + 1

        if (maxJ - minJ) % 2 != 0:
            if minJ > 0:
                minJ = minJ - 1
            elif maxJ < m-1:
                maxJ = maxJ + 1

        if dimensionsMaxes[GEOM_CENTER][0] < maxI - minI + 1:
            dimensionsMaxes[GEOM_CENTER] = (maxI - minI + 1, dimensionsMaxes[GEOM_CENTER][1])

        if dimensionsMaxes[GEOM_CENTER][1] < maxJ - minJ + 1:
            dimensionsMaxes[GEOM_CENTER] = (dimensionsMaxes[GEOM_CENTER][0], maxJ - minJ + 1)

        centersDict = {\
            GEOM_CENTER : ((maxI+minI)/2, (maxJ+minJ)/2),\
            RED_CENTER : (round(redSumI/redCount), round(redSumJ/redCount)),\
            MASS_CENTER : (round(centOfMassI/centOfMassCount), round(centOfMassJ/centOfMassCount)),\
            POND_CENTER : (round(pondSumI/pondSum), round(pondSumJ/pondSum))\
        }

        for (centerType, val) in filter(lambda x: x[0] != GEOM_CENTER, dimensionsMaxes.items()):
            dimensionsMaxes[centerType] =\
            getDimensions(\
                centersDict[GEOM_CENTER],\
                centersDict[centerType],\
                dimensionsMaxes[GEOM_CENTER][0],\
                dimensionsMaxes[GEOM_CENTER][1],\
                dimensionsMaxes[centerType][0],\
                dimensionsMaxes[centerType][1]\
            )

        centers.append(centersDict)



    for centerType in map(lambda pair: pair[0], dimensionsMaxes.items()):
        if centerType == POND_CENTER:
            continue
        trimmedData = []
        k = 0
        for picture in data:

            trimmedData.append((\
                trimImage(picture[0], centers[k][centerType], dimensionsMaxes[centerType][0], dimensionsMaxes[centerType][1]),\
                trimImage(picture[1], centers[k][centerType], dimensionsMaxes[centerType][0], dimensionsMaxes[centerType][1]),\
                trimImage(picture[2], centers[k][centerType], dimensionsMaxes[centerType][0], dimensionsMaxes[centerType][1]),\
                trimImage(picture[3], centers[k][centerType], dimensionsMaxes[centerType][0], dimensionsMaxes[centerType][1]),\
                picture[4]\
            ))

            k = k + 1

        comparablePixelsMax = []
        comparablePixelsMin = []

        k = 0
        for i in range(dimensionsMaxes[centerType][0]):
            comparablePixelsMax.append([])
            comparablePixelsMin.append([])
            for j in range(dimensionsMaxes[centerType][1]):

                flag = True
                for picture in trimmedData:
                    if i < 0 or i >= len(picture[3]) or j < 0 or j >= len(picture[3][0])\
                        or picture[3][i][j] == 0:
                        flag = False
                        break
                comparablePixelsMin[k].append(255 if flag else 0)


                flag = False
                for picture in trimmedData:
                    if not (i < 0 or i >= len(picture[3]) or j < 0 or j >= len(picture[3][0]))\
                        and picture[3][i][j] != 0:
                        flag = True
                        break

                comparablePixelsMax[k].append(255 if flag else 0)
            k = k + 1

        copyTrimmedData = deepcopy(trimmedData)

        for k in range(len(trimmedData)):
            for i in range(dimensionsMaxes[centerType][0]):
                for j in range(dimensionsMaxes[centerType][1]):
                    if not (i < 0 or i >= len(picture[3]) or j < 0 or j >= len(picture[3][0]))\
                        and comparablePixelsMax[i][j] == 0:
                        copyTrimmedData[k][0][i][j] = 0
                        copyTrimmedData[k][1][i][j] = 0
                        copyTrimmedData[k][2][i][j] = 0
                    if not (i < 0 or i >= len(picture[3]) or j < 0 or j >= len(picture[3][0]))\
                        and comparablePixelsMin[i][j] == 0:
                        trimmedData[k][0][i][j] = 0
                        trimmedData[k][1][i][j] = 0
                        trimmedData[k][2][i][j] = 0

            if not os.path.isdir(sys.argv[1] + "/rez/" + centerType + "/maxShape/"):
                os.makedirs(sys.argv[1] + "/rez/" + centerType + "/maxShape/")
            save_rgb_img(\
                np.asarray(copyTrimmedData[k][0]).transpose(),\
                np.asarray(copyTrimmedData[k][1]).transpose(),\
                np.asarray(copyTrimmedData[k][2]).transpose(),\
                sys.argv[1] + "/rez/" + centerType + "/maxShape/" + copyTrimmedData[k][4]\
            )

            if not os.path.isdir(sys.argv[1] + "/rez/" + centerType + "/minShape/"):
                os.makedirs(sys.argv[1] + "/rez/" + centerType + "/minShape/")
            save_rgb_img(\
                np.asarray(trimmedData[k][0]).transpose(),\
                np.asarray(trimmedData[k][1]).transpose(),\
                np.asarray(trimmedData[k][2]).transpose(),\
                sys.argv[1] + "/rez/" + centerType + "/minShape/" + trimmedData[k][4]\
            )
