import os
from glob import glob

def load_image_paths(base_dir):
    """
    base_dir: train 或 test 文件夹路径
    返回：[(path, class_index)] 或 [path]
    """
    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

    if len(subfolders) > 0:
        # 训练集: 每个类别一个文件夹
        paths = []
        classes = sorted(os.listdir(base_dir))
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        for cls in classes:
            img_paths = glob(os.path.join(base_dir, cls, "*.jpg"))
            paths += [(p, class_to_idx[cls]) for p in img_paths]

        return paths, classes
    else:
        # 测试集：图像平铺
        img_paths = glob(os.path.join(base_dir, "*.jpg"))
        return img_paths, None
