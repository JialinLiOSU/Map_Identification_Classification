# @author: Rui Li
# @date: 04/08/2019
# @comments: generate us map

# libaries ---------- basic
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from scipy.spatial import ConvexHull, Voronoi
import matplotlib.patches as mpatches


# libaries ---------- color scheme
# sequential
from matplotlib.cm import viridis
from matplotlib.cm import plasma
from matplotlib.cm import summer
from matplotlib.cm import gray
from matplotlib.cm import autumn
# diverging
from matplotlib.cm import PiYG
from matplotlib.cm import coolwarm
from matplotlib.cm import Spectral
from matplotlib.cm import BrBG
from matplotlib.cm import PRGn
# quantitative
from matplotlib.cm import Pastel1
from matplotlib.cm import Paired
from matplotlib.cm import Accent
from matplotlib.cm import Dark2
from matplotlib.cm import Set3

# NLTK libaraies, import corpus and extract frequent text
import nltk
from nltk.corpus import brown
nltk.download('brown')
nltk.download('universal_tagset')

# pandas, record the meta information
import pandas as pd

# define generation variables

# define generation variables

# configuration of map setting
isStateName = 0  # if show state name
isMainland = 1   # if show state that far from mainland: alaska
isRiver = 0      # if show
isCoastlines = 1 # if draw coastlines
isLat = 1        # if draw latitude
isLong = 1       # if draw longtitude
is3D = 0         # if draw 3D map

# configuration of map style
mapBackground = 0 # background color of the map
backgroundList = ['#FFFFFF', '#207ab0', '#333333', '#aedba8', '#fffdc4', '#a8d5e3', '#d1d1d1']
mapMask = 0   # whether use map mask, stylish layer
serviceList = ["ESRI_Imagery_World_2D",'Ocean_Basemap',"ESRI_StreetMap_World_2D",
               "NatGeo_World_Map","NGS_Topo_US_2D",
              "NGS_Topo_US_2D","World_Shaded_Relief", "World_Street_Map",
               "World_Topo_Map", "World_Terrain_Base", "canvas/World_Dark_Gray_Base",
               'Specialty/World_Navigation_Charts', 'Specialty/DeLorme_World_Base_Map',
               'Reference/World_Transportation', 'Reference/World_Reference_Overlay',
               'Canvas/World_Light_Gray_Base', 'World_Physical_Map'] # map style
colorList = ['#FFFFFF', '#F7C353', '#CDCDCD', '#f3A581']

# configuration of visualization variables
mapPosition = 0      # center point of map
mapSize = 0          # bounding box size of the map
mapProjection = 0    # projection of the map (can be regarded as shape)
mapValue = 1.0         # opacity of the color
mapColor = 0     # color (hue) scheme of map
mapOrientation = 0   # transformation of map
mapTexture = 0       # textures on the map
texturePatterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]

# configuration of visualization elements
mapText = 0    # random selected text
showLegend = 0  # show map color legend

meta_data = pd.read_csv('meta.csv', encoding='utf-8')

# us state name and acronym
short_state_names = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

# extract 100 sentence from Brown corpus with 'government' topics
brown_sent = brown.sents(categories='government')[0:2000]
brown_title = []
for i in range(len(brown_sent)):
    title = ' '.join(brown_sent[i]).split(',')[0]
    if(len(title) < 80 and len(title) > 20):
        brown_title.append(title)
brown_title = brown_title[0:100]

# extract top 100 frequent words from Brown corpus
noundist = nltk.FreqDist(w2 for ((w1, t1), (w2, t2)) in
            nltk.bigrams(brown.tagged_words(tagset="universal"))
            if w1.lower() == "the" and t2 == "NOUN")
frequent_dist = noundist.most_common(100)
frequent_words = []
for i in range(len(frequent_dist)):
    frequent_words.append(frequent_dist[i][0])


# parameters setting
def get_admin_level():
    a = random.uniform(0, 1)
    if (a <= 0.02):
        return 0
    else:
        return 1

# if show state name
def get_IsStateName():
    a = random.randint(1,3)
    if(a < 2):
        return 0
    elif(a == 2): # state name
        return 1
    elif(a == 3): # randomly generate text
        return 2

# if show far land like hawaii and alaska (only for us)
def get_IsMainland():
    a = random.uniform(0, 1)
    if (a <= 0.5):
        return 0
    else:
        return 1

    # if show lat or long

# if show lat and long
def getLatLong():
    a = random.uniform(0, 1)
    if (a <= 0.7):
        return 0, 0
    else:
        return 1, 1

    # get the background color of the map

def getBackgroundColor():
    a = random.randint(1, 17)
    if (a <= 6):
        return backgroundList[0]
    elif (6 < a <= 12):
        return backgroundList[1]
    else:
        b = a - 12
        return backgroundList[b]

# get the style of map
def getStyle():
    a = random.randint(0, 16)
    return serviceList[a]

# get the size
def getSize():
    a = random.randint(1, 8)
    if (a <= 2):
        return 0
    elif (2 < a <= 7):
        return 1
    else:
        return 2

# get the position
def getPosition(size):
    # small size
    if (size == 0):
        x1 = -random.randint(126, 129)
        x2 = -random.randint(41, 44)
        y1 = random.randint(6, 9)
        y2 = random.randint(47, 49)
        return x1, y1, x2, y2
    elif (size == 1):  # medium size
        x1 = -random.randint(118, 121)
        x2 = -random.randint(61, 65)
        y1 = random.randint(16, 20)
        y2 = random.randint(47, 51)
        return x1, y1, x2, y2
    else:  # large size
        x1 = -random.randint(117, 119)
        x2 = -random.randint(63, 65)
        y1 = random.randint(16, 20)
        y2 = random.randint(47, 51)
        return x1, y1, x2, y2

# get the position with style
def getStylePosition(size):
    # small size
    if(size == 0):
        x1 = -random.randint(136, 139)
        x2 = -random.randint(45, 50)
        y1 = random.randint(10, 15)
        y2 = random.randint(55, 60)
        return x1, y1, x2, y2
    elif(size == 1):  # medium size
        x1 = -random.randint(128, 131)
        x2 = -random.randint(60,65)
        y1 = random.randint(18, 24)
        y2 = random.randint(50, 55)
        return x1, y1, x2, y2
    else:  # large size
        x1 = -random.randint(127, 129)
        x2 = -random.randint(62, 66)
        y1 = random.randint(20, 24)
        y2 = random.randint(50, 52)
        return x1, y1, x2, y2

# get the value
def getValue():
    opcacity = round(random.uniform(0.7, 1), 2)
    return opcacity

# get the value
def getStyleValue():
    opcacity = round(random.uniform(0.0,0.2),1)
    return opcacity

# get the color scheme
# 0: blank 1-3: single color, 4-8: sequential color
# 9-13: diverging color 14-18: quantitative color
def getcolor_scheme():
    a = random.randint(0, 18)
    return a

# get color, i: random number, a: color scheme
def getColor(i, a):
    if(a == 0):
        return colorList[0]
    elif(1 <= a <= 3):
        return colorList[a]
    elif(4 <= a <= 8):
        i = (i + random.randint(0,2))*10
        if(a == 4):
            return viridis(i)
        elif(a == 5):
            return plasma(i)
        elif(a == 6):
            return summer(i)
        elif(a == 7):
            return gray(i)
        elif(a == 8):
            return autumn(i)
    elif(9 <= a <= 13):
        i = (i + random.randint(0,2))*20
        if(a == 9):
            return PiYG(i)
        elif(a == 10):
            return coolwarm(i)
        elif(a == 11):
            return Spectral(i)
        elif(a == 12):
            return BrBG(i)
        elif(a == 13):
            return PRGn(i)
    elif(13 < a <= 18):
        i = i - random.randint(0,4)
        if(i<0):
            return '#FFFFFF'
        if(a == 14):
            return Pastel1(i)
        elif(a == 15):
            return Paired(i)
        elif(a == 16):
            return Accent(i)
        elif(a == 17):
            return Dark2(i)
        elif(a == 18):
            return Set3(i)

# if add texture
def isTexture():
    a = random.randint(1, 4)
    if (a <= 3):
        return 0
    else:
        return 1

# generate texture
def getTexture():
    b = random.randint(0, 9)  # texture pattern
    c = random.randint(1, 3)  # density of texture pattern
    d = random.randint(0, 1)  # if a state marked with texture
    if (d == 0):
        return ''
    else:
        return texturePatterns[b] * c

# generate title
def getTitle():
    a = random.randint(0, 1)
    if (a == 0):
        return ''
    else:
        b = random.randint(0, 99)
        return brown_title[b]

# generate text on state
def getText():
    a = random.randint(0, 99)
    return frequent_words[a]

# generate legend
def getLegend(a):
    labels = random.sample(range(0, 99), 5)
    patch_1 = mpatches.Patch(color=getColor(1, a), label=frequent_words[labels[0]])
    patch_2 = mpatches.Patch(color=getColor(3, a), label=frequent_words[labels[1]])
    patch_3 = mpatches.Patch(color=getColor(5, a), label=frequent_words[labels[2]])
    patch_4 = mpatches.Patch(color=getColor(7, a), label=frequent_words[labels[3]])
    patch_5 = mpatches.Patch(color=getColor(9, a), label=frequent_words[labels[4]])
    return patch_1, patch_2, patch_3, patch_4, patch_5


# draw US map
def drawUSmap(index, filename):

    fig = plt.figure(figsize=(7, 5), dpi=150)

    # 1. size and location
    mapSize = getSize()
    x1, y1, x2, y2 = getPosition(mapSize)

    # map location and bounding box
    m = Basemap(llcrnrlon=x1, llcrnrlat=y1, urcrnrlon=x2, urcrnrlat=y2,
                projection='lcc', lat_1=33, lat_2=45, lon_0=-95)

    # 2. administraitive level
    admin_level = 1

    ax = plt.gca()  # get current axes instance

    # read polygon information from shape file, only show admin0 and admin1
    if(admin_level == 0):
        shp_info = m.readshapefile('data/us/USA_adm0', 'state', drawbounds=True, linewidth=0.5)

    else:
        shp_info = m.readshapefile('data/us/st99_d00', 'state', drawbounds=True, linewidth=0.5)
        # draw map
        # 3. color scheme
        colorscheme = getcolor_scheme()
        # 4. if show text on each state
        isStateName = get_IsStateName()
        # 5. identify the text size
        font_size = random.randint(4, 8)
        # 6. if add texture
        mapTexture = isTexture()
        # 7. if draw Alaska and Hawaii
        isMainland = get_IsMainland()
        # 8. identify the opacity value
        opaVal = getValue()
        printed_names = []
        for info, shape in zip(m.state_info, m.state):
            if(mapTexture == 1):
                poly = Polygon(shape, facecolor=getColor(len(info['NAME']), colorscheme),
                               edgecolor='k', alpha = opaVal, linewidth=0.5, hatch=getTexture())
            else:
                poly = Polygon(shape, facecolor=getColor(len(info['NAME']), colorscheme), alpha = opaVal, edgecolor='k', linewidth=0.5)

            if(isMainland == 1):
                if info['NAME'] == 'Alaska':
                    seg = list(map(lambda x, y: (0.35 * x + 1100000, 0.35 * y - 1300000), list(zip(*shape))[0],
                                   list(zip(*shape))[1]))
                    poly = Polygon(np.array(seg), facecolor=getColor(len(info['NAME']), colorscheme),
                                   alpha = opaVal, edgecolor='k', linewidth=0.5)
                if info['NAME'] == 'Hawaii':
                    seg = list(map(lambda x, y: (x + 5200000, y - 1400000), list(zip(*shape))[0], list(zip(*shape))[1]))
                    poly = Polygon(np.array(seg), facecolor=getColor(len(info['NAME']), colorscheme), edgecolor='k',
                                   alpha = opaVal, linewidth=0.5)

            ax.add_patch(poly)

            # add text on each state
            if(isStateName != 0):
                x, y = np.array(shape).mean(axis=0)
                hull = ConvexHull(shape)
                hull_points = np.array(shape)[hull.vertices]
                x, y = hull_points.mean(axis=0)
                short_name = list(short_state_names.keys())[list(short_state_names.values()).index(info['NAME'])]
                if short_name == 'AK' or short_name == 'HI' or short_name == 'PR': continue
                if short_name in printed_names: continue
                if(isStateName == 1):
                    plt.text(x + .1, y, short_name, ha="center", fontsize=font_size)
                elif(isStateName == 2):
                    state_text = getText()
                    plt.text(x + .1, y, state_text, ha="center", fontsize=font_size)
                printed_names += [short_name, ]

    # 9. if add long and lat
    isLat, isLong = getLatLong()
    if(isLat == 1):
        m.drawparallels(np.arange(25, 65, 20), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-120, -40, 20), labels=[0, 0, 0, 1])

    # 10. background color
    mapBackground = getBackgroundColor()
    ax.set_facecolor(mapBackground)

    # 11. if add title
    title = getTitle()
    plt.title(title)

    # 12. if add legends
    if(colorscheme >= 4):
        showLegend = 1
        loc_var = random.randint(1,5)
        if(loc_var == 1):
            p1, p2, p3, p4, p5 = getLegend(colorscheme)
            plt.legend(handles=[p1, p2, p3, p4, p5], loc='upper left', prop={'size': 6})
        elif(loc_var == 2):
            p1, p2, p3, p4, p5 = getLegend(colorscheme)
            plt.legend(handles=[p1, p2, p3, p4, p5], loc='upper right', prop={'size': 6})
        elif (loc_var == 3):
            p1, p2, p3, p4, p5 = getLegend(colorscheme)
            plt.legend(handles=[p1, p2, p3, p4, p5], loc='lower left', prop={'size': 6})
        elif (loc_var == 4):
            p1, p2, p3, p4, p5 = getLegend(colorscheme)
            plt.legend(handles=[p1, p2, p3, p4, p5], loc='lower right', prop={'size': 6})
        else:
            showLegend = 0
    else:
        showLegend = 0

    # remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # store the information into meta
    meta_data.loc[index, 'filename'] = filename
    meta_data.loc[index, 'country'] = 'USA'
    meta_data.loc[index, 'statename'] = isStateName
    meta_data.loc[index, 'mainland'] = isMainland
    meta_data.loc[index, 'lat and long'] = isLat
    meta_data.loc[index, 'background'] = mapBackground
    meta_data.loc[index, 'style'] = 'plain'
    meta_data.loc[index, 'position'] = str(x1) + ',' +  str(x2) + ',' + str(y1) + ',' + str(y2)
    meta_data.loc[index, 'size'] = mapSize
    meta_data.loc[index, 'projection'] = 'llc'
    meta_data.loc[index, 'opacity'] = opaVal
    meta_data.loc[index, 'color'] = colorscheme
    meta_data.loc[index, 'texture'] = mapTexture
    meta_data.loc[index, 'title'] = title
    meta_data.loc[index, 'legend'] = showLegend

    # plt.show()
    plt.savefig(filename)

# draw US map, admin0
def drawUSmapAdmin2(index, filename):

    fig = plt.figure(figsize=(7, 5), dpi=150)

    # 1. size and location
    mapSize = getSize()
    x1, y1, x2, y2 = getPosition(mapSize)

    # map location and bounding box
    m = Basemap(llcrnrlon=x1, llcrnrlat=y1, urcrnrlon=x2, urcrnrlat=y2,
                projection='lcc', lat_1=33, lat_2=45, lon_0=-95)

    # 2. administraitive level
    admin_level = 2

    ax = plt.gca()  # get current axes instance

    # read polygon information from shape file, only show admin0 and admin1
    if(admin_level == 2):
        shp_info = m.readshapefile('data/us/USA_adm2', 'state', drawbounds=True, linewidth=0.5)
        # draw map
        # 3. color scheme
        colorscheme = getcolor_scheme()
        # 4. if show text on each state
        isStateName = 0
        # 5. identify the text size
        font_size = random.randint(4, 8)
        # 6. if add texture
        mapTexture = isTexture()
        # 7. if draw Alaska and Hawaii
        isMainland = 0
        # 8. identify the opacity value
        opaVal = getValue()
        printed_names = []
        for info, shape in zip(m.state_info, m.state):
            if (mapTexture == 1):
                poly = Polygon(shape, facecolor=getColor(3, colorscheme),
                               edgecolor='k', alpha=opaVal, linewidth=0.5, hatch=getTexture())
            else:
                poly = Polygon(shape, facecolor=getColor(3, colorscheme), alpha=opaVal, edgecolor='k',
                               linewidth=0.5)

            if (isMainland == 1):
                if info['NAME'] == 'Alaska':
                    seg = list(map(lambda x, y: (0.35 * x + 1100000, 0.35 * y - 1300000), list(zip(*shape))[0],
                                   list(zip(*shape))[1]))
                    poly = Polygon(np.array(seg), facecolor=getColor(3, colorscheme),
                                   alpha=opaVal, edgecolor='k', linewidth=0.5)
                if info['NAME'] == 'Hawaii':
                    seg = list(map(lambda x, y: (x + 5200000, y - 1400000), list(zip(*shape))[0], list(zip(*shape))[1]))
                    poly = Polygon(np.array(seg), facecolor=getColor(len(info['NAME']), colorscheme), edgecolor='k',
                                   alpha=opaVal, linewidth=0.5)

            ax.add_patch(poly)

            # add text on each state
            if (isStateName != 0):
                x, y = np.array(shape).mean(axis=0)
                hull = ConvexHull(shape)
                hull_points = np.array(shape)[hull.vertices]
                x, y = hull_points.mean(axis=0)
                short_name = list(short_state_names.keys())[list(short_state_names.values()).index(info['NAME'])]
                if short_name == 'AK' or short_name == 'HI' or short_name == 'PR': continue
                if short_name in printed_names: continue
                if (isStateName == 1):
                    plt.text(x + .1, y, short_name, ha="center", fontsize=font_size)
                elif (isStateName == 2):
                    state_text = getText()
                    plt.text(x + .1, y, state_text, ha="center", fontsize=font_size)
                printed_names += [short_name, ]
    else:
        shp_info = m.readshapefile('data/us/st99_d00', 'state', drawbounds=True, linewidth=0.5)


    # 9. if add long and lat
    isLat, isLong = getLatLong()
    if(isLat == 1):
        m.drawparallels(np.arange(25, 65, 20), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-120, -40, 20), labels=[0, 0, 0, 1])

    # 10. background color
    mapBackground = getBackgroundColor()
    ax.set_facecolor(mapBackground)

    # 11. if add title
    title = getTitle()
    plt.title(title)

    # 12. if add legends
    if(colorscheme >= 4):
        showLegend = 0
        loc_var = random.randint(1,5)
    else:
        showLegend = 0

    # remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # store the information into meta
    meta_data.loc[index, 'filename'] = filename
    meta_data.loc[index, 'country'] = 'USA'
    meta_data.loc[index, 'statename'] = isStateName
    meta_data.loc[index, 'mainland'] = isMainland
    meta_data.loc[index, 'lat and long'] = isLat
    meta_data.loc[index, 'background'] = mapBackground
    meta_data.loc[index, 'style'] = 'plain'
    meta_data.loc[index, 'position'] = str(x1) + ',' +  str(x2) + ',' + str(y1) + ',' + str(y2)
    meta_data.loc[index, 'size'] = mapSize
    meta_data.loc[index, 'projection'] = 'llc'
    meta_data.loc[index, 'opacity'] = opaVal
    meta_data.loc[index, 'color'] = colorscheme
    meta_data.loc[index, 'texture'] = mapTexture
    meta_data.loc[index, 'title'] = title
    meta_data.loc[index, 'legend'] = showLegend

    #plt.show()
    plt.savefig('map/us_2/'+filename)

# draw US map, with style
def drawUSmapStyle(index, filename):

    fig = plt.figure(figsize=(7, 5), dpi=150)

    # 1. size and location
    mapSize = getSize()
    x1, y1, x2, y2 = getStylePosition(mapSize)

    # map location and bounding box
    m = Basemap(llcrnrlon=x1, llcrnrlat=y1, urcrnrlon=x2, urcrnrlat=y2,
                projection='lcc', lat_1=33, lat_2=45, lon_0=-95, epsg=3395)

    # 2. administraitive level
    admin_level = 1

    ax = plt.gca()  # get current axes instance

    # read polygon information from shape file, only show admin0 and admin1
    if(admin_level == 0):
        shp_info = m.readshapefile('data/us/USA_adm0', 'state', drawbounds=True, linewidth=0.2)

    else:
        shp_info = m.readshapefile('data/us/st99_d00', 'state', drawbounds=True, linewidth=0.03)
        # draw map
        # 3. color scheme
        colorscheme = getcolor_scheme()
        # 4. if show text on each state
        isStateName = 0
        # 5. identify the text size
        font_size = random.randint(4, 8)
        # 6. if add texture
        mapTexture = isTexture()
        # 7. if draw Alaska and Hawaii
        isMainland = 0
        # 8. identify the opacity value
        opaVal = getStyleValue()
        printed_names = []
        for info, shape in zip(m.state_info, m.state):
            if(mapTexture == 1):
                poly = Polygon(shape, facecolor=getColor(len(info['NAME']), colorscheme),
                               edgecolor='k', alpha = opaVal, linewidth=0.5, hatch=getTexture())
            else:
                poly = Polygon(shape, facecolor=getColor(len(info['NAME']), colorscheme), alpha = opaVal, edgecolor='k', linewidth=0.5)

            if(isMainland == 1):
                if info['NAME'] == 'Alaska':
                    seg = list(map(lambda x, y: (0.35 * x + 1100000, 0.35 * y - 1300000), list(zip(*shape))[0],
                                   list(zip(*shape))[1]))
                    poly = Polygon(np.array(seg), facecolor=getColor(len(info['NAME']), colorscheme),
                                   alpha = opaVal, edgecolor='k', linewidth=0.5)
                if info['NAME'] == 'Hawaii':
                    seg = list(map(lambda x, y: (x + 5200000, y - 1400000), list(zip(*shape))[0], list(zip(*shape))[1]))
                    poly = Polygon(np.array(seg), facecolor=getColor(len(info['NAME']), colorscheme), edgecolor='k',
                                   alpha = opaVal, linewidth=0.5)

            ax.add_patch(poly)

            # add text on each state
            if(isStateName != 0):
                x, y = np.array(shape).mean(axis=0)
                hull = ConvexHull(shape)
                hull_points = np.array(shape)[hull.vertices]
                x, y = hull_points.mean(axis=0)
                short_name = list(short_state_names.keys())[list(short_state_names.values()).index(info['NAME'])]
                if short_name == 'AK' or short_name == 'HI' or short_name == 'PR': continue
                if short_name in printed_names: continue
                if(isStateName == 1):
                    plt.text(x + .1, y, short_name, ha="center", fontsize=font_size)
                elif(isStateName == 2):
                    state_text = getText()
                    plt.text(x + .1, y, state_text, ha="center", fontsize=font_size)
                printed_names += [short_name, ]

    # 9. if add long and lat
    isLat, isLong = getLatLong()
    if(isLat == 1):
        m.drawparallels(np.arange(25, 65, 20), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-120, -40, 20), labels=[0, 0, 0, 1])

    # 10. background color
    mapBackground = getBackgroundColor()
    ax.set_facecolor(mapBackground)

    # 11. if add title
    title = getTitle()
    plt.title(title)

    # 12. if add legends
    if(colorscheme >= 4):
        showLegend = 1
        loc_var = random.randint(1,5)
        if(loc_var == 1):
            p1, p2, p3, p4, p5 = getLegend(colorscheme)
            plt.legend(handles=[p1, p2, p3, p4, p5], loc='upper left', prop={'size': 6})
        elif(loc_var == 2):
            p1, p2, p3, p4, p5 = getLegend(colorscheme)
            plt.legend(handles=[p1, p2, p3, p4, p5], loc='upper right', prop={'size': 6})
        elif (loc_var == 3):
            p1, p2, p3, p4, p5 = getLegend(colorscheme)
            plt.legend(handles=[p1, p2, p3, p4, p5], loc='lower left', prop={'size': 6})
        elif (loc_var == 4):
            p1, p2, p3, p4, p5 = getLegend(colorscheme)
            plt.legend(handles=[p1, p2, p3, p4, p5], loc='lower right', prop={'size': 6})
        else:
            showLegend = 0
    else:
        showLegend = 0

    # remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    mapStyle = getStyle()

    m.arcgisimage(service=mapStyle, xpixels=1000, verbose=True)

    # store the information into meta
    meta_data.loc[index, 'filename'] = filename
    meta_data.loc[index, 'country'] = 'USA'
    meta_data.loc[index, 'statename'] = isStateName
    meta_data.loc[index, 'mainland'] = isMainland
    meta_data.loc[index, 'lat and long'] = isLat
    meta_data.loc[index, 'background'] = mapBackground
    meta_data.loc[index, 'style'] = mapStyle
    meta_data.loc[index, 'position'] = str(x1) + ',' +  str(x2) + ',' + str(y1) + ',' + str(y2)
    meta_data.loc[index, 'size'] = mapSize
    meta_data.loc[index, 'projection'] = 'Mercator'
    meta_data.loc[index, 'opacity'] = opaVal
    meta_data.loc[index, 'color'] = colorscheme
    meta_data.loc[index, 'texture'] = mapTexture
    meta_data.loc[index, 'title'] = title
    meta_data.loc[index, 'legend'] = showLegend

    # plt.show()
    plt.savefig('map/us_2/' + filename)

# generate map image
def main():

    for i in range(len(meta_data)):
        if(i >= 980 and i < 1000):
            filename = 'us_map' + str(meta_data.loc[i,'id']) + '.png'
            drawUSmapAdmin2(i,filename)
    meta_data.to_csv('result.csv', index=False)

if __name__ == "__main__":    main()