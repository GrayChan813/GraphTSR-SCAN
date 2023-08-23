import os
import cv2
import json
import copy
from tqdm import tqdm

def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])

    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        # IoU = S_cross/(S1+S2-S_cross)
        IoU = S_cross/S2
        return IoU

if __name__ == "__main__":
    struc_root = 'checkpoints/SciTSR/Split_Res_50/json'
    img_root = 'datasets/SciTSR_Merge_test_v2/SciTSR_images_test'
    ocr_root = 'datasets/SciTSR_Merge_test/ocr'  # gt
    save_json_eval = struc_root.replace('json', 'eval')
    if not os.path.exists(save_json_eval): os.makedirs(save_json_eval)
    name_list = os.listdir(struc_root)
    for name in tqdm(name_list):
        # if name != '1502.06256v3.4.json': continue
        img = cv2.imread(os.path.join(img_root, name.replace('json', 'jpg')))
        img_r = img.copy()
        h,w,c = img_r.shape

        ocr_file = os.path.join(ocr_root, name.replace('json', 'txt'))
        with open(ocr_file, 'r') as f_ocr:
            ocr_list = f_ocr.readlines()

        struc_file = os.path.join(struc_root, name)
        with open(struc_file, 'r') as f_struc:
            struc_list = json.load(f_struc)
        
        # save_eval_json = copy.deepcopy(struc_list)
        # for idx in range(len(save_eval_json)):
        #     del save_eval_json[idx]['cell_points']
        for i in range(len(struc_list['cells'])):
            cell_pos_str = struc_list['cells'][i]['cell_points'].split(' ')
            cell_pos = list(map(int, cell_pos_str[0].split(',')))+list(map(int, cell_pos_str[2].split(',')))
            # cv2.rectangle(img_r, tuple(cell_pos[0:2]), tuple(cell_pos[2:4]), (255,0,0), 2)
            if cell_pos[0] == 0: cell_pos[0] = 20
            if cell_pos[1] == 0: cell_pos[1] = 20
            if cell_pos[2] >= w-20: cell_pos[2] == w-20
            if cell_pos[3] >= h-20: cell_pos[3] == h-20
            for j in range(len(ocr_list)):
                line = ocr_list[j].strip().split(' ')
                TL_pos = list(map(int, line[0].split(',')))
                text = line[1].strip()
                # cv2.rectangle(img_r, tuple(TL_pos[0:2]), tuple(TL_pos[2:4]), (0,255,0), 2)
                # cv2.imshow('img', img_r)
                # cv2.waitKey(0)
                IoU = compute_IOU(cell_pos, TL_pos)
                if IoU != 0:
                    struc_list['cells'][i]['tex'] = 'chenbangdong'
                    struc_list['cells'][i]['content'].append(text)
        json_file = os.path.join(save_json_eval, name)
        json.dump(struc_list, open(json_file, 'w'))
    print('DONE!')