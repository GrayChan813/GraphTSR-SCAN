import os
import cv2
import json
import shutil
from scitsr.eval import json2Relations, eval_relations


def example(gt_path, res_path):
    with open(gt_path) as fp:
        json_obj = json.load(fp)
    gt_relations = json2Relations(json_obj, splitted_content=True)

    with open(res_path) as fp:
        json_obj = json.load(fp)
    res_relations = json2Relations(json_obj, splitted_content=True)

    precision, recall = eval_relations(
        gt=[gt_relations], res=[res_relations], cmp_blank=True)
    f1 = 2.0 * precision * recall / \
        (precision + recall) if precision + recall > 0 else 0
    print("P: %.4f, R: %.4f, F1: %.4f" % (precision, recall, f1))


if __name__ == "__main__":
    error_list = []
    with open('datasets/ICDAR2013/test/error.txt') as f:
        for line in f.readlines():
            line = line.strip().replace('.png', '.json')
            error_list.append(line)

    # ===================================== 修改路径 ==================================
    ROOT = 'checkpoints/SciTSR/Split_Res_50'  # SciTSR
    gt_root = 'datasets/SciTSR_Merge_test/structure'  # SciTSR

    # ROOT = 'checkpoints/ICDAR2013/Split_Res_50'  # ICDAR2013
    # gt_root = 'datasets/ICDAR2013/test/json_new_eval_gt'  # ICDAR2013
    # ================================================================================

    res_root = os.path.join(ROOT, 'eval')
    im_root = os.path.join(ROOT, 'vis')

    save_error = os.path.join(ROOT, 'Error')
    if os.path.exists(save_error):
        shutil.rmtree(save_error)
    if not os.path.exists(save_error):
        os.makedirs(save_error)

    total_corr, total_res, total_gt = 0, 0, 0
    total_p_tmp, total_r_tmp, total_f1_tmp = 0, 0, 0
    name_list = os.listdir(res_root)
    f = open(os.path.join(ROOT, 'PRF.txt'), 'w')
    idx = 0
    for name in name_list:
        if name in error_list:
            continue
        print(name)

        gt_path = os.path.join(gt_root, name)
        res_path = os.path.join(res_root, name)
        try:
            with open(gt_path) as fp:
                json_obj = json.load(fp)
        except:
            import pdb
            pdb.set_trace()

        try:
            gt_relations = json2Relations(json_obj, splitted_content=True)
        except:
            continue

        with open(res_path) as fp:
            json_obj = json.load(fp)
        res_relations = json2Relations(json_obj, splitted_content=True)

        precision, recall, per_corr, per_res, per_gt = eval_relations(
            gt=[gt_relations], res=[res_relations], cmp_blank=False)
        f1 = 2.0 * precision * recall / \
            (precision + recall) if precision + recall > 0 else 0
        print("%s\t --- P: %.4f\t, R: %.4f\t, F1: %.4f" %
              (name, precision, recall, f1))

        total_corr, total_res, total_gt = total_corr + \
            per_corr, total_res+per_res, total_gt+per_gt
        total_p_tmp, total_r_tmp, total_f1_tmp = total_p_tmp + \
            precision, total_r_tmp+recall, total_f1_tmp+f1

        if f1 < 1:
            im = cv2.imread(os.path.join(im_root, name.replace('.json', '.jpg')), 1)  # SciTSR
            # im = cv2.imread(os.path.join(im_root, name.replace('.json', '.png')), 1)  # ICDAR2013
            cv2.imwrite(os.path.join(save_error, name.replace('json', 'jpg')), im)
        f.write("%s\t --- P: %.4f\t, R: %.4f\t, F1: %.4f\n" %
                (name, precision, recall, f1))
        idx += 1

    print("total nums: ", idx)

    # 以每张图片为单位计算指标
    total_p_tmp, total_r_tmp = total_p_tmp/idx, total_r_tmp/idx
    total_f1_tmp = 2.0 * total_p_tmp * total_r_tmp / \
        (total_p_tmp + total_r_tmp) if total_p_tmp + total_r_tmp > 0 else 0
    print("%s\t --- P: %.4f\t, R: %.4f\t, F1: %.4f" %
          ('(图片级别)total', total_p_tmp, total_r_tmp, total_f1_tmp))
    f.write("%s\t --- P: %.4f\t, R: %.4f\t, F1: %.4f\n" %
            ('(图片级别)total', total_p_tmp, total_r_tmp, total_f1_tmp))

    # 以每个邻接对为单位计算指标
    total_p = total_corr / total_res
    total_r = total_corr / total_gt
    total_f1 = 2.0 * total_p * total_r / \
        (total_p + total_r) if total_p + total_r > 0 else 0
    print("%s\t --- P: %.4f\t, R: %.4f\t, F1: %.4f" %
          ('(邻接对级别)total', total_p, total_r, total_f1))
    f.write("%s\t --- total_P: %.4f\t, total_R: %.4f\t, total_F1: %.4f\n" %
            ('(邻接对级别)total', total_p, total_r, total_f1))

    # 错误样本 -- 图片级别
    error_name_list = os.listdir(save_error)
    print("error nums: ", len(error_name_list))
    f.write("error nums: %d" % len(error_name_list))
    f.close()
