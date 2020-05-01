import argparse
import json
import os
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to data directory', default='./data/datasets')
parser.add_argument('--filename', type=str, help='Filename for sample', default='medication-train')
parser.add_argument('--out_dir', type=str, help='Path to output file dir', default='./data/datasets')
parser.add_argument('--seed', type=int, help='random seed', default=55)
parser.add_argument('--sample_ratio', type=float, default=0.2,
                    help='how many questions are sampled from the original dataset')


def main(args):
    t0 = time.time()

    in_file = os.path.join(args.data_dir, args.filename + '.json')
    print('Loading dataset %s' % in_file)
    dataset = json.load(open(in_file, 'r'))
    print("Sampling...")

    new_data = subsample_dataset_random(dataset, args.sample_ratio, args.seed)

    out_file = os.path.join(args.out_dir, '%s-sampled-%s' % (args.filename, str(args.sample_ratio)) + '.json')
    print('Will write to file %s' % out_file)
    json.dump(new_data, open(out_file, 'w'))

    print('Total time: %.4f (s)' % (time.time() - t0))


def subsample_dataset_random(data_json, sample_ratio=0.01, seed=55):
    new_data = []
    total = 0
    sample = 0
    random.seed(seed)
    for paras in data_json['data']:
        new_paragraphs = []
        for para in paras['paragraphs']:
            new_qas = []
            context = para['context']
            qa_num = len(para['qas'])
            total += qa_num
            sample_num = int(qa_num * sample_ratio)
            sampled_list = [i for i in range(qa_num)]
            sampled_list = random.choices(sampled_list, k=sample_num)
            for qa_id in sampled_list:
                qa = para['qas'][qa_id]
                sample += 1
                new_qas.append(qa)
            new_para = {'context': context, 'qas': new_qas}
            new_paragraphs.append(new_para)
        new_data.append({'title': paras['title'], 'paragraphs': new_paragraphs})
    new_data_json = {'data': new_data, 'version': data_json['version']}
    print('Total QA Num: %d, Sample QA Num: %d' % (total, sample))
    return new_data_json


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
