# importing os module
import os
 
# Function to rename multiple files
def main():
    dirName = r'D:\OneDrive - The Ohio State University\Images for training\maps for classification of projections\Other_Projections_Maps\online_temp'
    prefix = 'online_other_'
    temp_images = os.listdir(dirName)
    for fileName in temp_images:
        os.rename(dirName + "/" + fileName, dirName + "/" + prefix + fileName)

# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()