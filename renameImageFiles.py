
# Python 3 code to rename multiple
# files in a directory or folder
 
# importing os module
import os
 
# Function to rename multiple files
def main():
    folder = r"D:\OneDrive - The Ohio State University\Map classification\Images for training\projectionRecognition\EqualArea_Projection_Maps\onlineMaps"
    for count, filename in enumerate(os.listdir(folder)):
        postFix = filename.split('.')[-1]
        # online_robinson_projection_map2.jpg
        dst = 'online_equalArea_projection_map_' + str(count) + '.' + postFix
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{dst}"
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()