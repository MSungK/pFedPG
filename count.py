import os
import os.path as osp


def get_files_count(path):
    cnt = 0
    for file in os.listdir(path):
        if osp.isdir(osp.join(path,file)):
            cnt += get_files_count(osp.join(path, file))
        else:
            cnt += 1
    return cnt


if __name__ == '__main__':
    path = 'dataset/office_caltech_10'
    tmp = ['caltech', 'amazon', 'webcam', 'dslr']
    cnt = 0 
    for t in tmp:
        tmp_path = osp.join(path, t)
        cnt += get_files_count(tmp_path)
    print(cnt)