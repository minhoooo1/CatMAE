import os
from PIL import Image
from tqdm import tqdm

def resize_images_in_folder(folder_path, H=480, W=880):
    
    
    # 定义目标路径
    A_target_folder = os.path.join(folder_path, "Annotations", "480p")

    # 遍历目标文件夹下的所有子文件夹
    for subdir in tqdm(os.listdir(A_target_folder)):
        subdir_path = os.path.join(A_target_folder, subdir)

        # 确保这是一个文件夹
        if os.path.isdir(subdir_path):
            # 遍历子文件夹中的所有图片
            for filename in os.listdir(subdir_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(subdir_path, filename)
                    
                    # 打开和调整图像大小
                    img = Image.open(img_path)
                    resized_img = img.resize((W, H), Image.NEAREST)

                    # 保存调整大小后的图像
                    resized_img.save(img_path)
    
    
    
    J_target_folder = os.path.join(folder_path, "JPEGImages", "480p")

    # 遍历目标文件夹下的所有子文件夹
    for subdir in tqdm(os.listdir(J_target_folder)):
        subdir_path = os.path.join(J_target_folder, subdir)

        # 确保这是一个文件夹
        if os.path.isdir(subdir_path):
            # 遍历子文件夹中的所有图片
            for filename in os.listdir(subdir_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(subdir_path, filename)
                    
                    # 打开和调整图像大小
                    img = Image.open(img_path)
                    resized_img = img.resize((W, H), Image.NEAREST)

                    # 保存调整大小后的图像
                    resized_img.save(img_path)

    
    
    work_path = os.getcwd()
    fa = open("davis_vallist_480_880.txt", "a") 
    print("\n###writing\n")
    with open(os.path.join(folder_path, "ImageSets", "2017", "val.txt"), "r") as f:    
        val_img_list = f.readlines()
        for cate in val_img_list:
            cate = cate.strip()
            if cate:
                print(work_path + "/" + J_target_folder + "/" + cate + " " + work_path + "/" + A_target_folder + "/" + cate )
                fa.write(work_path + "/" + J_target_folder + "/" + cate + " " + work_path + "/" + A_target_folder + "/" + cate + "\n")
    
    f.close()
    
# 使用
resize_images_in_folder('data/DAVIS_480_880')
