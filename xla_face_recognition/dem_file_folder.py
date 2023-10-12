import os
path = "/home/maicg/Desktop/database_local/face"
name_txt_save = "cout.txt"

def count_files_in_folder(folder_path):
    count = 0
    for _, _, files in os.walk(folder_path):
        count += len(files)
    return count
with open(name_txt_save, "w") as file:
    for i in range(734):
        sub_folder_name = f"ID{i}"
        path_folder = os.path.join(path, sub_folder_name)
        rs_count = count_files_in_folder(path_folder)
        file.write(str(rs_count) + "\n")
