import os

data_path = 'C:\\Users\\chy\\Desktop\\dense\\images\\val'
img_names = os.listdir(data_path)

list_file = open('data/shipval.txt', 'w')
for img_name in img_names:
    list_file.write('C:/Users/chy/Desktop/dense/images/val/%s\n' % img_name)

list_file.close()