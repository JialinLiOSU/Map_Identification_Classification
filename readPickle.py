# horizonStreets and verticalStreets contain results from text recognition
import pickle

# with open('selectedRoadInterCoordPairs.pkl', 'rb') as file:
#     # A new file will be created
#     selectedRoadInterCoordPairs= pickle.load(file)

# for roadInterName in selectedRoadInterCoordPairs.keys():
#     print('roadInterName: ' + roadInterName)
#     print(selectedRoadInterCoordPairs[roadInterName])

path = r'C:\Users\jiali\Desktop\Map_Identification_Classification'
pickleFile = 'imgNameList_after_shuffle_region1500.pickle'
with open(path + '\\' + pickleFile, 'rb') as file:
    # A new file will be created
    data = pickle.load(file)
testImages = data[1200:]
print(testImages)
print('test')