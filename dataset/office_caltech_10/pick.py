import pickle
import os

path = '/workspace/Minsung/vpt/dataset/domain_net/DomainNet'
files = os.listdir(path)
cnt = 0
for file in files:
    with open(os.path.join(path,file),"rb") as fr:
        data = pickle.load(fr)
        cnt += len(data[0])
print(cnt)