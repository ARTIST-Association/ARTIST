import os
import main
import re

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles



config_folder = "Results\\test\\"
folders = next(os.walk(config_folder))[1]

todo_index = [re.search("_*", folder).end() for folder in folders]
todo_folders = [folder for i, folder in enumerate(folders) if todo_index[i] == 0]
# todo_folders = [r"_*" in folder for folder in folders]
# for folder in folders:
#     todo_folders.append(re.finditer("_*",folder))
print(todo_folders)
# experiment_names = getListOfFiles(config_folder)
# print(experiment_names)

# for experiment_name in experiment_names:
#     main.main(experiment_name)
    