# pickle image data for carto regino classification
import pickle

data_path = r'D:\OneDrive - The Ohio State University\cartoRegioForVGG'

with open(data_path + '/' + 'carto_region_test_10.pickle', 'rb') as file:
    [x_test, y_test] = pickle.load(file)

with open(data_path + '/' + 'imgNameList_carto_10.pickle', 'rb') as file:
    imgNameList = pickle.load(file)
print('test')