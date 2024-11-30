import numpy as np
import math
import transbigdata as tbd
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
import os
import time
import random


def N(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))


def getBrightColor():
    # 获得亮色，保证其中两色分别90和ff，第三色为任意值即可
    full_range = ["0", "1", "2", "3", "4", "5", "6",
                  "7", "8", "9", "a", "b", "c", "d", "e", "f"]
    combination = ["90"]
    # 将 ff 随机插入 90 前或后
    combination.insert(random.randint(0, 1), "ff")
    third_color = "{0}{1}".format(
        full_range[random.randint(0, 15)], full_range[random.randint(0, 15)]
    )
    combination.insert(random.randint(0, 2), third_color)
    color = "#" + "".join(combination)
    return color


def LatLng2Degree(LatZero, LngZero, Lat, Lng):
    """
    Args:
        point p1(latA, lonA)
        point p2(latB, lonB)
    Returns:
        bearing between the two GPS points,
        default: the basis of heading direction is north
    """
    radLatA = math.radians(LatZero)
    radLonA = math.radians(LngZero)
    radLatB = math.radians(Lat)
    radLonB = math.radians(Lng)
    dLon = radLonB - radLonA
    y = math.sin(dLon) * math.cos(radLatB)
    x = math.cos(radLatA) * math.sin(radLatB) - \
        math.sin(radLatA) * math.cos(radLatB) * math.cos(dLon)
    brng = math.degrees(math.atan2(y, x))
    brng = (brng + 360) % 360
    return brng


def getlinestringlen(linestring_geom):
    ln = [k[0] for k in list(linestring_geom.coords)]
    lt = [k[1] for k in list(linestring_geom.coords)]
    temp = pd.DataFrame()
    temp['lon1'] = ln
    temp['lat1'] = lt
    temp['lon2'] = temp['lon1'].shift(-1)
    temp['lat2'] = temp['lat1'].shift(-1)
    temp = temp.iloc[:-1]
    length = (tbd.getdistance(temp['lon1'], temp['lat1'],
                              temp['lon2'], temp['lat2'])).sum()
    return length

    """
    linestring =  [k[::-1] for k in list(linestring_geom.coords)]
    length=0
    while len(linestring)>1:
        pt1=linestring.pop()
        pt2=linestring.pop()
        length += geodesic(pt1,pt2).m
    if len(linestring) == 1:
        length += geodesic(pt2,linestring.pop()).m 
    return length
    """
