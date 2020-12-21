# read the shapefiles of cartograms 
# to get the coordinate extent of each region

"""
Find out the states with the specified relation with one state
The relations include adjacency, distance and orientation relation
"""


import numpy as np
import pickle
import sys
sys.path.append(r'C:\Users\jiali\Desktop\Map_Identification_Classification\world map generation\getCartoCoordExtent')
from shapex import *

def getExtent(shp, country):
    for c in shp:
        if c['properties']['CNTRY_NAME'] == country:
            break
    typeGeom = c['geometry']['type']
    coordGeom = c['geometry']['coordinates']
    minLat,maxLat, minLon, maxLon= 999, -999, 999, -999
    # if typeGeom != 'MultiPolygon':
    #     coordGeom = [coordGeom]
    
    for poly in coordGeom:
        if typeGeom != 'MultiPolygon':
            poly = [poly]
        tmpMinLon, tmpMaxLon = min(poly[0])[0], max(poly[0])[0]
        tmpMinLat, tmpMaxLat = min(poly[0], key = lambda t: t[1])[1], max(poly[0],key = lambda t: t[1])[1]
        if tmpMinLon < minLon:
            minLon = tmpMinLon
        if tmpMaxLon > maxLon:
            maxLon = tmpMaxLon
        if tmpMinLat < minLat:
            minLat = tmpMinLat
        if tmpMaxLat > maxLat:
            maxLat = tmpMaxLat

    return minLat,maxLat, minLon, maxLon

def getUSExtent(shp, country):
    for c in shp:
        if c['properties']['CNTRY_NAME'] == country:
            break
    typeGeom = c['geometry']['type']
    coordGeom = c['geometry']['coordinates']
    minLat,maxLat, minLon, maxLon= 999, -999, 999, -999
    
    poly = coordGeom[0]
    if typeGeom != 'MultiPolygon':
        poly = [poly]
    # print(poly[0])
    tmpMinLon, tmpMaxLon = min(poly[0])[0], max(poly[0])[0]
    tmpMinLat, tmpMaxLat = min(poly[0], key = lambda t: t[1])[1], max(poly[0],key = lambda t: t[1])[1]
    if tmpMinLon < minLon:
        minLon = tmpMinLon
    if tmpMaxLon > maxLon:
        maxLon = tmpMaxLon
    if tmpMinLat < minLat:
        minLat = tmpMinLat
    if tmpMaxLat > maxLat:
        maxLat = tmpMaxLat

    return minLat,maxLat, minLon, maxLon


if __name__ == "__main__":
    # path of the cartogram shapefiles
    shapefilePath = r'C:\Users\jiali\Desktop\Map_Identification_Classification\world map generation\shpfile\cartogram\CylindricalEqualAreaWorld'
    numIterList = [10,15,20,25,30,40,55,70,85,100]
    numIterListUS = [5, 8, 11, 14, 17, 20, 23, 26, 29, 32]
    fileNameList = ['cartogram_POP2007_iter_' + str(num) + '_WGS84.shp' for num in numIterList]
    countryList = ['China', 'South Korea', 'United States']

    extentList = []
    for fileName in fileNameList:
        shp = shapex(shapefilePath + '\\' + fileName)
        country = countryList[1]
        extent = getExtent(shp, country)
        print(fileName+','+country)
        print(extent)



