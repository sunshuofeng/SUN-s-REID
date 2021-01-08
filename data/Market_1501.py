import torch
import torch.utils.data as Data
from fastai.vision.all import get_image_files
import os
from PIL import Image


class Market_1501:
    def __init__(self,root):
        super(Market_1501, self).__init__()
        self.train_dir=os.path.join(root,'bounding_box_train')
        self.gallery_dir=os.path.join(root,'bounding_box_test')
        self.query_dir=os.path.join(root,'query')
        self.train=self.make_data(self.train_dir)
        self.query=self.make_data(self.query_dir)
        self.gallery=self.make_data(self.gallery_dir)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_data_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_data_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_data_info(self.gallery)



    def get_data_info(self,datas):
        pids=set()
        cids=set()
        image_counts=len(datas)
        for data in datas:
            pid=data[1]
            cid=data[2]
            pids.add(pid)
            cids.add(cid)
        return image_counts,len(pids),len(cids)



    def make_data(self,path):




        names=get_image_files(path)

        pids=set()
        for name in names:
            name = str(name)
            name = name.split('/')[-1]
            pid = name.split('_')[0]
            if pid == '-1':
                continue
            else:
                pids.add(pid)
        pid_to_label={pid:label for label,pid in enumerate(pids)}


        datas=[]
        for name in names:
            name=str(name)
            path=name
            name=name.split('/')[-1]
            pid=name.split('_')[0]
            if pid=='-1':
              continue
            else:
                pid=pid_to_label[pid]
            cid=int(name.split('_')[1][1])
            datas.append((path,pid,cid))
        return datas



