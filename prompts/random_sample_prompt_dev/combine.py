import os
from tqdm import tqdm
def main():
    path_prefix = "./results/codeparrot/codeparrot-temp1.0-len512-k40/internet"
    path_save = "./results/codeparrot/codeparrot-temp1.0-len512-k40/internet/ffffffffffffffffffffffffffffffffffffffff"
    os.makedirs(path_save)
    # get all the results from the folder
    file_folder_hash = os.listdir(path_prefix)
    # go through all the folders
    file_number = 0
    for file_folder in tqdm(file_folder_hash,total=len(file_folder_hash)):
        # get the files in the folder
        files = os.listdir(os.path.join(path_prefix,file_folder))
        # go through all the files
        for f in files:
            if f == 'prompt.txt':
                continue
            # copy the file to antoher folder, and rename it
            os.system("cp {} {}".format(os.path.join(path_prefix,file_folder,f),os.path.join(path_save,"{}".format(file_number))))
            file_number += 1

if __name__ == "__main__":
    main()