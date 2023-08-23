import cv2 as cv
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from matplotlib import patches


def isInterArea(testPointx, testPointy, AreaPoint):  # testPoint为待测点[x,y]
    LBPoint = AreaPoint[6:8]  # AreaPoint为按顺时针顺序的4个点[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    LTPoint = AreaPoint[0:2]
    RTPoint = AreaPoint[2:4]
    RBPoint = AreaPoint[4:6]
    testPoint=[0,0]
    testPoint[0] = testPointx
    testPoint[1] = testPointy
    a = (LTPoint[0] - LBPoint[0]) * (testPoint[1] - LBPoint[1]) - (LTPoint[1] - LBPoint[1]) * (
            testPoint[0] - LBPoint[0])
    b = (RTPoint[0] - LTPoint[0]) * (testPoint[1] - LTPoint[1]) - (RTPoint[1] - LTPoint[1]) * (
            testPoint[0] - LTPoint[0])
    c = (RBPoint[0] - RTPoint[0]) * (testPoint[1] - RTPoint[1]) - (RBPoint[1] - RTPoint[1]) * (
            testPoint[0] - RTPoint[0])
    d = (LBPoint[0] - RBPoint[0]) * (testPoint[1] - RBPoint[1]) - (LBPoint[1] - RBPoint[1]) * (
            testPoint[0] - RBPoint[0])
    # print(a,b,c,d)
    if (a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0):
        return True
    else:
        return False

# print ('isInterArea',isInterArea(30,47,[10,10,150,10,150,50,10,50]))

class BBox(object):
    '''
    BBox 类
    接受 bbox 和 index作为BBox编号
    类中含有 context 变量 用于存放该bbox的文本行识别结果，请自助添加
    '''

    def __init__(self, bbox, index):
        self.bbox = bbox.copy()
        self.x1 = max(bbox[0].copy(), 0)
        self.y1 = max(bbox[1].copy(), 0)

        self.x2 = max(bbox[2].copy(), 0)
        self.y2 = max(bbox[3].copy(), 0)

        self.x3 = max(bbox[4].copy(), 0)
        self.y3 = max(bbox[5].copy(), 0)

        self.x4 = max(bbox[6].copy(), 0)
        # print(bbox[7])
        self.y4 = max(bbox[7].copy(), 0)

        # self.bb= bbox.copy()

        self.x1p = max(bbox[0].copy(), 0)
        self.y1p = max(bbox[1].copy(), 0)

        self.x2p = max(bbox[2].copy(), 0)
        self.y2p = max(bbox[3].copy(), 0)

        self.x3p = max(bbox[4].copy(), 0)
        self.y3p = max(bbox[5].copy(), 0)

        self.x4p = max(bbox[6].copy(), 0)
        self.y4p = max(bbox[7].copy(), 0)

        # self.w = bbox[2]
        # self.h = bbox[3]
        #
        # self.x2 = self.x1 + self.w
        # self.y2 = self.y1
        #
        # self.x3=self.x2
        # self.y3 = self.y1 + self.h
        #
        # self.x4 = self.x1
        # self.y4 = self.y3



        # self.x2 = self.x1 + self.w
        # self.y2 = self.y1 + self.h

        self.index = index
        self.rowspan = 0
        self.colspan = 0
        self.cols = []
        self.rows = []
        # self.context = ""
        #self.text = text

        self.xmids = self.getxmids()
        self.ymidz = self.getymidz()
        self.xmidx = self.getxmidx()
        self.ymidy = self.getymidy()


        self.printed = 0
        # self.source_z_x = (self.x1 + self.x3 ) / 4
        # self.source_z_y = (self.y1 + self.y2 + self.y3 + self.y4 ) / 4
        self.source_z_x = (self.x1 + self.x2 + self.x3 + self.x4 ) / 4
        self.source_z_y = (self.y1 + self.y2 + self.y3 + self.y4 ) / 4
        self.source_zp_x = (self.x1p + self.x2p + self.x3p + self.x4p ) / 4
        self.source_zp_y = (self.y1p + self.y2p + self.y3p + self.y4p) / 4



    def getxmids(self):
        return (self.x1p + self.x2p)/2

    def getymidz(self):
        return (self.y1p + self.y4p)/2

    def getxmidx(self):
        return (self.x3p + self.x4p)/2

    def getymidy(self):
        return (self.y3p + self.y2p)/2

        


class BBox2HTML(object):
    '''
    接受含有 bbox 的 list

    BBoxtoHTML 方法 返回html生成结果
    该结果默认不带文本，若要加入文本请自助编写代码给生成的BBox类的context变量赋值

    '''
    def __init__(self, bboxs, name):
        self.BBoxList1 = list()
        self.BBoxList2 = list()
        self.BBoxList = list()
        self.bboxdict = dict()
        self.name = name
        # self.rowspandict = dict()
        # self.colspandict = dict()

        i = 0
        for bbox in bboxs:
            self.BBoxList1.append(BBox(bbox, i))
            self.BBoxList2.append(BBox(bbox, i))
            self.BBoxList.append(BBox(bbox, i))

            self.bboxdict[str(bbox)] = i
            i += 1

    def EmptyCellSearching(self):
        #todo
        pass

    def refine(self,rowmatched):
        self.BBoxList1 = list()
        num =0
        min_r,max_r = 1e5, 0 
        for row in rowmatched:
            for item in row:  
                min_r = min(min_r,(item[0]+item[6])/2)
                max_r = max(max_r, (item[2]+item[4])/2)
        for row in rowmatched:
            if (row[0][0]>min_r and row[0][0]<min_r-100):row[0][0] = min_r
            if (row[0][6]>min_r and row[0][6]<min_r-100):row[0][6] = min_r
            if (row[-1][2]<max_r and row[-1][2]>max_r+100):row[-1][2] = max_r
            if (row[-1][4]<max_r and row[-1][4]>max_r+100):row[-1][4] = max_r
            for i in range(len(row)):
                self.bboxdict[str(row[i])] = num
                self.BBoxList1.append(BBox(row[i],num))
                num+=1

    def BBoxtoHTML(self):

        rowmatched = self.HorizontallyCellMatching1()
        colmatched = self.VerticallyCellMatching1()

        return rowmatched, colmatched

    def HorizontallyCellMatching1(self):
        all = self.BBoxList2
        linelist = list()
        indexlist = list()  
        all = sorted(all, key=lambda x: x.source_z_y, reverse=False) 
      
        for bbox in all:
            line = list()
            indexs = list()
            bbz = list()
            bby = list()
            bbi = list()
            z_cc = list()
            y_cc = list()
            line.append(bbox.bbox)
            indexs.append(bbox.index)
            bbi.append(bbox)
            bbz.append(bbox)
            bby.append(bbox)
            EXIST_FLAG = 0

            if (True):
                    source = bbz[0]
                    pi = 0
                    num = 0
                    
                    for target1 in all:
                        if((source.y1 + source.y2) / 2 < (target1.y1 + target1.y2)/ 2-10 and (source.y4+source.y3) / 2 > (target1.y4 + target1.y3) / 2 + 10):
                            continue         
                        if target1.source_z_x <= source.source_z_x:
                            if self.horizontallyconnected(source, target1) and self.horizontallyconnected_top(source, target1) and self.horizontallyconnected_bot(source, target1):
                                num = num+1
                                if source is not target1:
                              
                                    w = (source.x1p + source.x4p)/2 - (target1.x3p + target1.x2p)/2
                                    if isInterArea((source.x1p + source.x4p) / 2 - w - 5, (source.y1p + source.y4p) / 2,
                                            target1.bbox) or isInterArea((source.x1p + source.x4p) / 2 - w - 10,
                                                (source.y1p + source.y4p) / 2, target1.bbox) or isInterArea(
                                            (source.x1p + source.x4p) / 2 - w, (source.y1p + source.y4p) / 2, target1.bbox):
                                            bbi.append(target1)
                                            source.y1p= max(source.y1p, target1.y2-3)
                                            source.y4p= min(source.y4p, target1.y3+3)

                        elif target1.source_z_x > source.source_z_x:
                                    num = num+1
                                    w = (target1.x1p+ target1.x4p)/2 - (source.x3p + source.x2p)/2
                                    if self.horizontallyconnected1(source, target1) and self.horizontallyconnected_top1(source, target1) and self.horizontallyconnected_bot1(source, target1):
                                        if source is not target1:
                                            if isInterArea((source.x2p + source.x3p) / 2 +w + 5, (source.y2p + source.y3p) / 2,
                                                target1.bbox) or isInterArea((source.x2p + source.x3p) / 2 +w + 10,
                                                    (source.y2p + source.y3p) / 2, target1.bbox) or isInterArea(
                                                (source.x2p + source.x3p) / 2 +w , (source.y2p + source.y3p) / 2,target1.bbox):
                                                    bbi.append(target1)
                                                    source.y2p = max(source.y2p,target1.y1-3)
                                                    source.y3p = min(source.y3p,target1.y4+3)

           
            for ii in bbi:
                if ii.index not in indexs:
                    line.append(ii.bbox)
                    indexs.append(ii.index)

            indexs.sort()

            if len(indexlist) > 0:
                for existed in indexlist:                           
                    
                    if existed == indexs:     
                        EXIST_FLAG = 1
                        break
                # print(EXIST_FLAG)
                if not EXIST_FLAG:
                    linelist.append(line)    
                    indexlist.append(indexs) 
            else:
                linelist.append(line)    
                indexlist.append(indexs)

        # print(indexlist)
        sortedlinelist = sorted(linelist, key=lambda x: np.array(x)[:, 1].mean()+np.array(x)[:, 3].mean()+np.array(x)[:, 5].mean()+np.array(x)[:, 7].mean(), reverse=False)
        rowmatched = list()
        for line in sortedlinelist:
            line = sorted(line, key=lambda x: x[0], reverse=False)
            for i in range(len(line)):
                line[i] = tuple(line[i])
            rowmatched.append(line)

        rowmatched_index = list()
        for line in indexlist:
            line = tuple(line)
            rowmatched_index.append(line)

        rowmatched = list(filter(lambda f: not any(set(f) < set(g) for g in rowmatched), rowmatched))
        rowmatched_index = list(filter(lambda f: not any(set(f) < set(g) for g in rowmatched_index), rowmatched_index))

        for row in rowmatched:
            for i in range(len(row)):
                row[i] = list(row[i])

        for row in rowmatched_index:
            row = list(row)

        adj = []
        for line in rowmatched_index:
            for i in range(len(line) - 1):
                if line[i] < line[i + 1]:
                    adj.append([line[i], line[i + 1]])
                else:
                    adj.append([line[i + 1], line[i]])

        # print(np.array(adj).T)
        return np.array(adj).T
        # return rowmatched


    def cross_match(self): 
        all=self.BBoxList1
        linelist = list()
        indexlist = list()
        all = sorted(all, key=lambda x: (x.source_z_y,x.source_z_x), reverse=False)
        for bbox in all:
            line = list()
            indexs = list()
            bbi = list()
            EXIST_FLAG=0
            # print('bbox',bbox.bbox)
            # if(bbox.bbox!=[1335.0, 222.0, 1554.0, 222.0, 1554.0, 277.0, 1335.0, 277.0]):continue
            # removelist = list()
            line.append(bbox.bbox)
            indexs.append(bbox.index)
            bbi.append(bbox)
            source = bbox
            for target1 in all:
                # print("targets:",target1.bbox)

                # print((target1.x1+target1.x4)/2,(target1.x2+target1.x3)/2,(source.x1+source.x4)/2,(source.x2+source.x3)/2)
                cross_flag =(target1.y4+target1.y3)/2-8<(source.y1+source.y2)/2  or (
                            (target1.y2+target1.y1)/2+8 >(source.y3+source.y4)/2) 
                # print(cross_flag)
                if(cross_flag):
                    
                    if((source.x1+source.x4)/2>(target1.x1+target1.x4)/2-5 and (source.x2+source.x3)/2<(target1.x2+target1.x3)/2+5):
                        bbi.append(target1)
                        # continue
                    cross_flagz =(target1.x2+target1.x3)/2-3 >(source.x1+source.x4)/2  and (
                                        (target1.x2+target1.x3)/2+3 <(source.x2+source.x3)/2) and (
                                        (target1.x1+target1.x4)/2+3<(source.x1+source.x4)/2)
                    # print(cross_flagz)
                    if(cross_flagz):
                        x_rel1 = (target1.x2) - (source.x1)
                        x_source1 = (source.x2 - (source.x1))
                        # x_source1 = ((source.x2+source.x3)/2 - (source.x1+source.x4)/2)
                        x_rel2 = (target1.x3) - (source.x4)
                        x_source2 = ((source.x3) - (source.x4))
                        p1 = x_rel1/(x_source1)
                        p2 = x_rel2/(x_source2)
                        # print(p1,x_rel1,x_source1)
                        # print(p2,x_rel2,x_source2)
                        if(p1>0.25 and p2 >0.25):       #0.25
                            bbi.append(target1)
                    
                            
            for ii in bbi:
                if ii.index not in indexs:
                    line.append(ii.bbox)
                    indexs.append(ii.index)
            indexs.sort()

            if len(indexlist) > 0:
                # print('len(indexlist) >0')
                for existed in indexlist:

                    if existed == indexs:
                        EXIST_FLAG = 1
                        break

                if not EXIST_FLAG:
                    linelist.append(line)
                    indexlist.append(indexs)
            else:
                linelist.append(line)
                indexlist.append(indexs)   

        for bbox in all:
            line = list()
            indexs = list()
            bbi = list()
            EXIST_FLAG = 0
            line.append(bbox.bbox)
            indexs.append(bbox.index)
            bbi.append(bbox)
            source = bbox
            for target1 in all:
                cross_flag =(target1.y4+target1.y3)/2-8 <(source.y1+source.y2)/2  or (
                            (target1.y2+target1.y1)/2+8 >(source.y3+source.y4)/2) 
                # print(cross_flag)
                if(not cross_flag):continue
                if((source.x1+source.x4)/2>(target1.x1+target1.x4)/2-5 and (source.x2+source.x3)/2<(target1.x2+target1.x3)/2+5):
                    bbi.append(target1)
                    # continue
                cross_flagy =(target1.x1+target1.x4)/2+3 <(source.x2+source.x3)/2  and (
                        (target1.x1+target1.x4)/2 -3>(source.x1+source.x4)/2)and (
                        (target1.x2+target1.x3)/2 -3>(source.x2+source.x3)/2)
                # print(cross_flagy)
                if(cross_flagy):
                    x_rel1 = (source.x2) - (target1.x1)
                    # x_source = ((source.x2+source.x3)/2 - (source.x1+source.x4)/2)
                    x_source1 = ((source.x2) - (source.x1))
                    x_rel2 = (source.x3) - (target1.x4)
                    x_source2 = ((source.x3) - (source.x4))
                    p1 = x_rel1/(x_source1)
                    p2 = x_rel2/(x_source2)
                    # print(p,x_rel,x_source)
                    if(p1>0.25 and p2>0.25):     #simple 0.3      #inclined 0.25
                        bbi.append(target1)
                        
            for ii in bbi:
                if ii.index not in indexs:
                    line.append(ii.bbox)
                    indexs.append(ii.index)

            
            indexs.sort()
            # print('line',line)
            # print('indexs',indexs)

            if len(indexlist) > 0:
                # print('len(indexlist) >0')
                for existed in indexlist:
                    if existed == indexs:
                        EXIST_FLAG = 1
                        break
                # print(EXIST_FLAG)
                if not EXIST_FLAG:
                    linelist.append(line)
                    indexlist.append(indexs)
            else:
                linelist.append(line)
                indexlist.append(indexs)
        return linelist, indexlist

    def VerticallyCellMatching1(self):
        all=self.BBoxList1
        linelist = list()
        indexlist = list()
        all = sorted(all, key=lambda x: (x.source_z_y,x.source_z_x), reverse=False)
              
        for bbox in all:
            line = list()
            indexs = list()
            bbs = list()
            bbx = list()
            bbi = list()
            s_cc = list()
            x_cc = list()
            # removelist = list()
            line.append(bbox.bbox)
            indexs.append(bbox.index)
            bbi.append(bbox)
            bbs.append(bbox)
            bbx.append(bbox)
            
            # removelist.append(bbox)
            EXIST_FLAG = 0
            lenn=len(all)
            # print(lenn)
            # for j in range(lenn):
            #     if len(bbs) == 0:
            #         break
            #     else:
            if(True):
                source = bbs[0]
                pi=0

                for target1 in all:
                    if((source.x1+source.x4)/2<(target1.x1+target1.x4)/2-5 and (source.x2+source.x3)/2>(target1.x2+target1.x3)/2+5):
                            continue    

                    if target1.source_z_y<source.source_z_y: #and  (source.y1p + source.y2p)/2 > (target1.y3p+ target1.y4p)/2-6:
                        
                        if self.verticallyconnected(source, target1) and self.verticallyconnected_left(source, target1) and self.verticallyconnected_right(source, target1):
                            # print("targets:",target1.bbox)
                            if source is not target1:
                                h = (source.y1p + source.y2p)/2 - (target1.y3p+ target1.y4p)/2  
        
                                if isInterArea((source.x1p + source.x2p) / 2 , (source.y1p + source.y2p) / 2 -h- 5,
                                        target1.bbox) or isInterArea((source.x1p + source.x2p) / 2 , (source.y1p + source.y2p) / 2 -h-10,
                                        target1.bbox) or isInterArea((source.x1p + source.x2p) / 2 , (source.y1p + source.y2p) / 2 -h-15,
                                        target1.bbox) or isInterArea((source.x1p + source.x2p) / 2 , (source.y1p + source.y2p) / 2 -h,
                                        target1.bbox):
                                    bbi.append(target1)
                                    source.x1p= max(source.x1p,target1.x4-3)
                                    source.x2p= min(source.x2p,target1.x3+3)

                    else:   
                    # elif (target1.y1p + target1.y2p)/2  > (source.y3p+ source.y4p)/2-6:
                        if self.verticallyconnected1(source, target1) and self.verticallyconnected_left1(source, target1) and self.verticallyconnected_right1(source, target1):
                              # print("targetx:",target1.bbox)
                                h = (target1.y1p + target1.y2p)/2 - (source.y3p+ source.y4p)/2  
                                # print(h,(source.x4p + source.x3p) / 2 ,(source.y3p + source.y4p) / 2 +h)
                                if isInterArea((source.x4p + source.x3p) / 2 , (source.y3p + source.y4p) / 2 +h+5,
                                        target1.bbox) or isInterArea((source.x4p + source.x3p) / 2 , (source.y3p + source.y4p) / 2 +h+ 10,
                                        target1.bbox) or isInterArea((source.x4p + source.x3p) / 2 , (source.y3p + source.y4p) / 2 +h+ 15,
                                        target1.bbox) or isInterArea((source.x4p + source.x3p) / 2 , (source.y3p + source.y4p) / 2 +h,
                                        target1.bbox):
                                    bbi.append(target1)
                                    source.x3p= max(source.x3p,target1.x1-3)
                                    source.x4p= min(source.x4p,target1.x2+3)
                    
            for ii in bbi:
                if ii.index not in indexs:
                    line.append(ii.bbox)
                    indexs.append(ii.index)
            indexs.sort()

            if len(indexlist) > 0:
                for existed in indexlist:

                    if existed == indexs:
                        EXIST_FLAG = 1
                        break
                # print(EXIST_FLAG)
                if not EXIST_FLAG:
                    linelist.append(line)
                    indexlist.append(indexs)
            else:
                linelist.append(line)
                indexlist.append(indexs)

        linelist_cross, index_cross = self.cross_match()
        for i in range(len(linelist_cross)):
            EXIST_FLAG=0
            indexs = index_cross[i]
            line = linelist_cross[i]
            if len(indexlist) > 0:
                for existed in indexlist:

                    if existed == indexs:
                        EXIST_FLAG = 1
                        break
                # print(EXIST_FLAG)
                if not EXIST_FLAG:
                    linelist.append(line)
                    indexlist.append(indexs)
            else:
                linelist.append(line)
                indexlist.append(indexs)

        # print(linelist)
        # print(indexlist)
        sortedlinelist = sorted(linelist, key=lambda x: np.array(x)[:, 0].mean(), reverse=False)
        colmatched = list()
        for line in sortedlinelist:
            line = sorted(line, key=lambda x: x[1], reverse=False)
            for i in range(len(line)):
                line[i] = tuple(line[i])
            colmatched.append(line)

        colmatched_index = list()
        for line in indexlist:
            line = tuple(line)
            colmatched_index.append(line)

        colmatched = list(filter(lambda f: not any(set(f) < set(g) for g in colmatched), colmatched))
        colmatched_index = list(filter(lambda f: not any(set(f) < set(g) for g in colmatched_index), colmatched_index))

        # print(colmatched)
        # print(colmatched_index)
        for col in colmatched:
            for i in range(len(col)):
                col[i] = list(col[i])

        for col in colmatched_index:
            col = list(col)

        # print(colmatched)
        # print(colmatched_index)

        adj = []
        for line in colmatched_index:
            for i in range(len(line) - 1):
                if line[i] < line[i + 1]:
                    adj.append([line[i], line[i + 1]])
                else:
                    adj.append([line[i + 1], line[i]])

        # print(np.array(adj).T)
        return np.array(adj).T
        # return colmatched

    #zuo
    def horizontallyconnected(self, source, target):
        # print((source.y1p+source.y4p)/2, target.y2, target.y3)
        if (source.y1p + source.y4p) / 2 >= target.y2 + 2 and (source.y1p + source.y4p) / 2 <= target.y3 - 2:
            return True
        else:
            return False
    def horizontallyconnected_top(self, source, target):
        # print("top:",(source.y1p + source.y4p) /2 - (source.y4p-source.y1p)/8, target.y2, target.y3)
        if (source.y1p + source.y4p) /2 - (source.y4p-source.y1p)/8 >= target.y2 +2 and (source.y1p + source.y4p) /2 - (source.y4p-source.y1p)/8 <= target.y3 - 2:
            return True
        else:
            return False
    def horizontallyconnected_bot(self, source, target):
        # print("bot:",(source.y1p+source.y4p)/2+ (source.y4p-source.y1p)/8, target.y2, target.y3)
        if (source.y1p + source.y4p)/2  + (source.y4p-source.y1p)/8 >= target.y2 +2 and (source.y1p + source.y4p)/2 + (source.y4p-source.y1p)/8<= target.y3 - 2:
            return True
        else:
            return False

    def horizontallyconnected1(self, source, target):
        # print((source.y2p+source.y3p)/2, target.y1, target.y4)
        if (source.y2p + source.y3p) / 2 >= target.y1  and (source.y2p + source.y3p) / 2 <= target.y4 :
            return True
        else:
            return False
    def horizontallyconnected_top1(self, source, target):
        # print((source.y2p+source.y3p)*11.0/32, target.y1, target.y4)
        if (source.y2p + source.y3p) /2 - (source.y3p-source.y2p)/8 >= target.y1+2  and (source.y2p + source.y3p) /2- (source.y3p-source.y2p)/8 <= target.y4-2:
            return True
        else:
            return False
    def horizontallyconnected_bot1(self, source, target):
        # print((source.y2p+source.y3p)*13.0/32, target.y1, target.y4)
        if (source.y2p + source.y3p)/2+ (source.y3p-source.y2p)/8 >= target.y1+2  and (source.y2p + source.y3p)/2+ (source.y3p-source.y2p)/8 <= target.y4-2:
            return True
        else:
            return False
    #shang
    def verticallyconnected(self, source, target):
        # print((source.x1p+source.x2p)/2, target.x4, target.x3)
        if (source.x1p+source.x2p)/2 >= target.x4+3 and (source.x1p+source.x2p)/2 <= target.x3-3:
            return True
        else:
            return False
    def verticallyconnected_left(self, source, target):
        # print((source.x1p+source.x2p)/2, target.x4, target.x3)
        if (source.x1p+source.x2p)/2 - (source.x2p-source.x1p)/16  >= target.x4+3 and (source.x1p+source.x2p)/2-(source.x2p-source.x1p)/16 <= target.x3-3:
            return True
        else:
            return False
    def verticallyconnected_right(self, source, target):
        # print((source.x1p+source.x2p)/2, target.x4, target.x3)
        if (source.x1p+source.x2p)/2 + (source.x2p-source.x1p)/16 >= target.x4+3 and (source.x1p+source.x2p)/2 +(source.x2p-source.x1p)/16 <= target.x3-3:
            return True
        else:
            return False

    def verticallyconnected1(self, source, target):
        # print((source.x3p+source.x4p)/2, target.x1, target.x2)
        if (source.x3p+source.x4p)/2 >= target.x1+3 and (source.x3p+source.x4p)/2 <= target.x2-3:
            return True
        else:
            return False
    def verticallyconnected_left1(self, source, target):
        # print((source.x3p+source.x4p)/2, target.x1, target.x2)
        if (source.x3p+source.x4p)/2 -(source.x3p-source.x4p)/16 >= target.x1+3 and (source.x3p+source.x4p)/2 - (source.x3p-source.x4p)/16<= target.x2-3:
            return True
        else:
            return False
    def verticallyconnected_right1(self, source, target):
        # print((source.x3p+source.x4p)/2, target.x1, target.x2)
        if (source.x3p+source.x4p)/2 + (source.x3p-source.x4p)/16 >= target.x1+3 and (source.x3p+source.x4p)/2 + (source.x3p-source.x4p)/16 <= target.x2-3:
            return True
        else:
            return False






