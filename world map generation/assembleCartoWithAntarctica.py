from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
# from mpl_toolkits.basemap import Basemap
# from matplotlib.patches import Polygon
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import PathPatch
# import matplotlib.patches as mpatches

path = 'C:\\Users\\li.7957\\Desktop\\Map_Identification_Classification\\world map generation\\'
shpFileName = 'shpfile/cartogram/worldCartoWithAntarctica/carto_pop_15_wgs84'

# draw world map

def assembleCartoImages(index, filename):

    # check aspect ratio
    asp_x = random.randint(7, 8)
    asp_y = random.randint(4, 5)

    fig = plt.figure(figsize=(12, 4), dpi=500)

    # 1. size and location
    mapSize = getSize()

    y1, y2, x1, x2 = 3276907, 5188175, -12925569, -7853600
    deltaX = x2 - x1
    deltaY = y2 - y1

    # map location and bounding box
    m = Basemap(lon_0=0, 
                projection='cea', fix_aspect=True)
    # m.drawcoastlines(linewidth=0.25)
    # m.drawcountries(linewidth=0.25)

    # 2. administraitive level
    admin_level = 0

    ax = plt.gca()  # get current axes instance
    # ax = plt.Axes(fig, [0.25, 0.25, 0.75, 0.75], )

    # read polygon information from shape file, only show admin0 and admin1
    if (admin_level == 0):
        shp_info = m.readshapefile(
            path + shpFileName, 'state', drawbounds=True, linewidth=0.01)
        # 3. color scheme
        colorscheme = getcolor_scheme()
        # 4. if show text on each state
        isStateName = get_IsStateName()
        # 5. identify the text size
        font_size = random.randint(1, 2)
        # font_size = 0.5
        # 6. if add texture # no textures needed
        # mapTexture = isTexture()
        # 7. if draw Alaska and Hawaii
        isMainland = 1
        # 8. identify the opacity value
        opaVal = getValue()
        printed_names = []
        for info, shape in zip(m.state_info, m.state):
            # if (mapTexture == 1):
            #     poly = Polygon(shape, facecolor=getColor(len(info['CNTRY_NAME']), colorscheme),
            #                    edgecolor='k', alpha=opaVal, linewidth=0.5, hatch=getTexture())
            # else:
            poly = Polygon(shape, facecolor=getColor(len(info['NAME']), colorscheme),
                               alpha=opaVal, edgecolor='k', linewidth=0.05)

            ax.add_patch(poly)

            # add text on each state
            isStateName = 0
            if (isStateName != 0):
                x, y = np.array(shape).mean(axis=0)
                hull = ConvexHull(shape)
                hull_points = np.array(shape)[hull.vertices]
                x, y = hull_points.mean(axis=0)
                short_name = info['CNTRY_NAME']
                if short_name in printed_names:
                    continue
                if (isStateName == 1):
                    plt.text(x + .1, y, short_name,
                             ha="center", fontsize=font_size)
                # elif (isStateName == 2):
                #     state_text = getText()
                #     plt.text(x + .1, y, state_text,
                #              ha="center", fontsize=font_size)
                printed_names += [short_name, ]

    # draw map

    # 9. if add long and lat
    isLat, isLong = getLatLong()
    # if (isLat == 1):
    #     margin = random.randint(2, 4) * 10
    #     m.drawparallels(np.arange(-90, 90, margin), labels=[1, 0, 0, 0], linewidth=0.2, fontsize=5)
    #     m.drawmeridians(np.arange(-180, 180, margin), labels=[0, 0, 0, 1], linewidth=0.2, fontsize=5)

    # 10. background color
    mapBackground = getBackgroundColor()
    ax.set_facecolor(mapBackground)

    # store the information into meta
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.savefig(path+filename)
    # plt.close()
    original = Image.open(path+filename)
    width, height = original.size   # Get dimensions

    extentMinX, extentMaxX = -19116870.2084505856037140, 19472718.4939954914152622
    extentMinY, extentMaxY = -4865081.7273503318428993,6350581.8445226745679975
    left = (x1 - (extentMinX)-deltaX/20)/(extentMaxX-extentMinX)  * width + 200 # standard cyl
    top = (extentMaxY - y2 - deltaY/20   ) / (extentMaxY-extentMinY) *height 
    right = (x2 - (extentMinX)+deltaX/20 )/(extentMaxX-extentMinX) * width +200
    bottom = (extentMaxY - y1 + deltaY/20   ) / (extentMaxY-extentMinY) * height 
    croppedImage = original.crop((left, top, right, bottom))

    croppedImage.save(path+filename)

    img = mpimg.imread(path+filename)
    fig = plt.figure(dpi=150)
    ax = plt.gca()  # get current axes instance
    # fig = plt.figure(figsize=(asp_x, asp_y), dpi=150)
    imgplot = plt.imshow(img)


    # remove borders
    plt.axis('off')
    plt.savefig(path+filename, bbox_inches='tight')
    plt.close()
    plt.show()

# draw world map with style
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def main():
    savingPath = r"C:\Users\jiali\Desktop\Map_Identification_Classification\world map generation"
    
    # get the part of antartica
    antar_path = r"C:\Users\jiali\Desktop\Map_Identification_Classification\world map generation"
    antarcticaImage = 'carto_world_antarctica_0_cea_0.png'

    original = Image.open(antar_path + "\\" + antarcticaImage)
    width, height = original.size   # Get dimensions
    left = 0
    top = height*9/10
    right = width
    bottom = height
    antarctica = original.crop((left, top, right, bottom))
    antarctica.show()

    # get the part of world without antartica
    carto_path = r""
    carto_images = os.listdir(carto_path)
    for cartoImageName in carto_images:
        original_carto = Image.open(antar_path + "\\" + antarcticaImage)
        width, height = original_carto.size
        left = 0
        top = 0
        right = width
        bottom = height*9/10
        carto = original.crop((left, top, right, bottom))
        get_concat_v(carto, antarctica).save(savingPath + '\\' + cartoImageName)
        
    # get_concat_h(rightImage, leftImage).save(path+filename)
    # # get_concat_v(im1, im1).save('data/dst/pillow_concat_v.jpg')

    # img = mpimg.imread(path+filename)
    # fig = plt.figure(dpi=150)
    # ax = plt.gca()  # get current axes instance
    # # fig = plt.figure(figsize=(asp_x, asp_y), dpi=150)
    # imgplot = plt.imshow(img)

if __name__ == "__main__":
    main()
