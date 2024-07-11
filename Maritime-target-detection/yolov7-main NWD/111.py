from PIL import Image
import os

# 设置要修改的图片文件夹路径
folder_path = 'C:\\Users\\chy\\Desktop\\113'

# 设置目标图片大小
target_size = (164, 96)

# 遍历文件夹中的所有图片文件
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 打开图片文件
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)

        # 修改图片大小并保存
        resized_image = image.resize(target_size)
        resized_image.save(image_path)
print("批量操作完成！")