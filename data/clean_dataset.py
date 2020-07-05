import os
image_names = os.listdir('/home/liuyun/dataset/UTKFace')
for label in image_names:
    age, gender, race = list(map(int, label.split('.')[0].split('_')[:3]))
    if not ((-1 < age < 117)and (gender in [0, 1]) and (race in [0, 1, 2, 3, 4])):
        print(age, gender, race)
        print(label)