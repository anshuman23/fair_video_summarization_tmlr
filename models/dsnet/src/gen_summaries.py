import logging
from pathlib import Path

import numpy as np
import torch
import argparse
import pickle
import os

from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from modules.model_zoo import get_model

logger = logging.getLogger()


def evaluate(model, val_loader, nms_thresh, device):
    model.eval()
    # stats = data_helper.AverageMeter('fscore', 'diversity')

    with torch.no_grad():
        all_summaries = []
        fscores = []
        for test_key, seq, _, cps, n_frames, nfps, picks, user_summary, _ in val_loader:
            # print(test_key)

            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

            pred_cls, pred_bboxes = model.predict(seq_torch)

            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

            pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)
            pred_summ = vsumm_helper.bbox2summary(
                seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)

            # print(pred_summ.shape)
            # print(user_summary.shape)
            all_summaries.append(pred_summ)

            eval_metric = 'avg' if 'tvsum' or 'fvs' in test_key else 'max'
            fscore = vsumm_helper.get_summ_f1score(
                pred_summ, user_summary, eval_metric)

            fscores.append(fscore)
            # print(f'{test_key} : {fscore}')
            # pred_summ = vsumm_helper.downsample_summ(pred_summ)
            # diversity = vsumm_helper.get_summ_diversity(pred_summ, seq)
            # stats.update(fscore=fscore, diversity=diversity)
            # stats.update(fscore=fscore, diversity=0)
    # exit()
    print('Avg fscore: ', sum(fscores)/len(fscores))
    return all_summaries


def main(args):
    # args = init_helper.get_arguments()

    # init_helper.init_logger(args.model_dir, args.log_file)
    # init_helper.set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(vars(args))
    model = get_model(args.model, **vars(args))
    model = model.eval().to(device)

    # for split_path in args.splits:
    splits = data_helper.load_yaml(args.split_path)

    # stats = data_helper.AverageMeter('fscore', 'diversity')

    state_dict = torch.load(str(args.ckpt_path),
                                map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    split = splits[args.split]

    val_set = data_helper.VideoDataset(split['test_keys'])
    val_loader = data_helper.DataLoader(val_set, shuffle=False)

    all_summaries = evaluate(model, val_loader, args.nms_thresh, device)

    split_dir = os.path.join(args.out_dir, f'split{args.split}')
    os.makedirs(split_dir, exist_ok=True)
    with open(f'{split_dir}/summaries.pkl', 'wb') as file:
        pickle.dump(all_summaries, file)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate summaries for a specific split using a trained model")
    parser.add_argument("--model", type=str, default='anchor-free', choices=('anchor-based', 'anchor-free'))
    parser.add_argument('--base-model', type=str, default='attention',
                        choices=['attention', 'lstm', 'linear', 'bilstm',
                                 'gcn'])
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-hidden', type=int, default=128)
    parser.add_argument('--nms-thresh', type=float, default=0.5)
    
    parser.add_argument("--ckpt_path", type=str, help="Path to the trained model file")
    # parser.add_argument("--dataset", default="../../datasets/fvs.h5",type=str, help="Path to the h5 dataset")
    parser.add_argument("--split_path", type=str, default="../splits/fvs.yml", help="Path to the splits file")
    parser.add_argument("--split", type=int, default=0, help="ID of the split to save summaries for")
    parser.add_argument("--out_dir", type=str, default="models/fvs/summaries",help="Directory to save the generated summaries")
    args = parser.parse_args()
    main(args)
