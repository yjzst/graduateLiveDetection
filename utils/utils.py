import os
import numpy as np
import pandas as pd


def Image2CSV(realpath,fakepath,csvpath):
    real = os.listdir(realpath)
    real.sort(key=lambda x:int(x[:-4]))
    fake = os.listdir(fakepath)
    fake.sort(key=lambda x:int(x[:-4]))
    real = [realpath+"\\"+i for i in real]
    fake = [fakepath + "\\" + i for i in fake]
    total = real + fake
    total = np.array(total)
    image_count_data = pd.DataFrame(total)
    image_count_data.to_csv(csvpath,index=False,header=True)
    real_len = len(real)
    fake_len = len(fake)
    return real_len,fake_len
def Label2CSV(real_len,fake_len,csv_path):
    label_0 = []
    label_1 = []
    df = pd.read_csv(csv_path)
    for i in range(real_len):
        label_0.append(0)
    for j in range(fake_len):
        label_1.append(1)
    total = label_0 + label_1
    c = pd.DataFrame(total)
    df[1] = c
    print(df)
    df.to_csv(csv_path,header=False,index=None)

def Face2CSV(path1,path2,path3,csv):
    makangzhe = os.listdir(path1)
    yangjiezhi = os.listdir(path2)
    yanglinxi = os.listdir(path3)
    ma = [path1 + "\\" + i for i in makangzhe]
    yang1 = [path2 + "\\" + i for i in yangjiezhi]
    yang2 = [path3 + "\\" + i for i in yanglinxi]
    total = ma + yang1 + yang2
    total = np.array(total)
    image_count_data = pd.DataFrame(total)
    image_count_data.to_csv(csv, index=False, header=False)


if __name__ == '__main__':
    # realpath = r"D:\liveness_detection\graduateLiveDetection\data\face-antispoof-data\real"
    # fakepath = r"D:\liveness_detection\graduateLiveDetection\data\face-antispoof-data\fake"
    # csvpath = r"D:\liveness_detection\graduateLiveDetection\data\429Data.csv"
    # real_len, fake_len = Image2CSV(realpath, fakepath, csvpath)
    # Label2CSV(real_len, fake_len, csvpath)
    path1 = r"D:\liveness_detection\graduateLiveDetection\data\student\Makangzhe"
    path2 = r"D:\liveness_detection\graduateLiveDetection\data\student\Yangjiezhi"
    path3 = r"D:\liveness_detection\graduateLiveDetection\data\student\Yanglinxi"
    csv = r"D:\liveness_detection\graduateLiveDetection\data\student\student.csv"
    Face2CSV(path1, path2, path3, csv)