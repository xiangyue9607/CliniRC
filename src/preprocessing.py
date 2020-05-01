import argparse
import json
import os
import random
import re
import time
import uuid

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to data directory', default='./data/datasets')
parser.add_argument('--filename', type=str, help='emrQA dataset filename', default='data.json')
parser.add_argument('--out_dir', type=str, help='Path to output file dir', default='./data/datasets')
parser.add_argument('--seed', type=int, help='random seed', default=55)
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='how many questions are used for training')
parser.add_argument('--dev_ratio', type=float, default=0.1,
                    help='how many questions are used for validation')
parser.add_argument("--max_answer_length", type=int, default=20,
                    help="filter QAs whose answer length (# of words) exceeds the max len")


def main(args):
    t0 = time.time()

    print("begin cleaning...")
    input_file = os.path.join(args.data_dir, args.filename)
    new_json = clean_datasets(filename=input_file)
    print("Splitting the dataset into Medication and Relation subsets...")
    subsets = split_datasets_into_subsets(new_json, max_answer_length=args.max_answer_length)
    subset_name = ['medication', 'relation']
    for i, subset in enumerate(subsets):
        show_dataset_statistics(subset, subset_name[i])
        split_train_dev_test_per_note(subset, args.out_dir, args.train_ratio, args.dev_ratio, subset_name[i])

    print('Total time: %.4f (s)' % (time.time() - t0))


def prune_sentence(sent):
    sent = " ".join(sent.split())
    # replace html special tokens
    sent = sent.replace("_", " ")
    sent = sent.replace("&lt;", "<")
    sent = sent.replace("&gt;", ">")
    sent = sent.replace("&amp;", "&")
    sent = sent.replace("&quot;", "\"")
    sent = sent.replace("&nbsp;", " ")
    sent = sent.replace("&apos;", "\'")
    sent = sent.replace("_", "")
    sent = sent.replace("\"", "")

    # remove whitespaces before punctuations
    sent = re.sub(r'\s([?.!\',)"](?:\s|$))', r'\1', sent)
    # remove multiple punctuations
    sent = re.sub(r'[?.!,]+(?=[?.!,])', '', sent)
    sent = " ".join(sent.split())
    return sent


def clean_datasets(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    new_json = dict()
    new_json["version"] = "1.0"
    new_json["data"] = list()
    # Data level
    for i in range(len(data["data"])):
        actual_data = data["data"][i]
        if actual_data['title'] not in ["medication", "relations"]:
            continue
        new_data = dict()
        new_data["title"] = actual_data["title"]
        new_data["paragraphs"] = list()
        # Paragraph level
        for j in tqdm(range(len(actual_data["paragraphs"]))):
            new_paragraph = dict()
            temp_list = actual_data["paragraphs"][j]["context"]
            # remove whitespaces between texts and punctuations
            context = " ".join((" ".join(temp_list).split()))
            context = prune_sentence(context)
            new_paragraph["context"] = context
            new_paragraph["qas"] = list()
            qas = actual_data["paragraphs"][j]["qas"]
            if len(qas) == 0:
                continue
            # QAS Level
            for k in range(len(qas)):
                new_qas = dict()
                new_qas["answers"] = list()
                answers = qas[k]["answers"]
                # Answers level
                for l in range(len(answers)):
                    if answers[l]["answer_entity_type"] == 'complex':
                        continue
                    new_answers = dict()
                    if actual_data['title'] == 'risk':
                        evidence = answers[l]['text']
                    else:
                        evidence = answers[l]["evidence"]
                    evidence = prune_sentence(evidence)
                    if evidence:
                        while evidence[-1] in [',', '.', '?', '!', '-', ' ']:
                            evidence = evidence[:-1]
                        char_pos = -1
                        temp_evidence = evidence
                        final_evidence = temp_evidence
                        num = 0
                        while char_pos == -1:
                            char_pos = context.find(temp_evidence)
                            final_evidence = temp_evidence
                            temp_evidence = ' '.join(temp_evidence.split()[:-1])
                            num += 1
                        if char_pos > 0 and final_evidence:
                            new_answers["answer_start"] = char_pos
                            new_answers["text"] = final_evidence
                            new_qas["answers"].append(new_answers)
                        else:
                            continue
                questions = qas[k]["question"]
                new_answers = new_qas['answers']
                if len(new_answers) == 0:
                    continue
                for p in range(len(questions)):
                    new_qas = dict()
                    new_qas["question"] = questions[p]
                    # generate a unique uuid for each question (similar to SQuAD)
                    new_qas["id"] = str(uuid.uuid1().hex)
                    new_qas['answers'] = new_answers
                    new_paragraph["qas"].append(new_qas)

            new_data["paragraphs"].append(new_paragraph)
        new_json["data"].append(new_data)

    return new_json


def split_datasets_into_subsets(dataset_json, max_answer_length=20):
    # split the dataset into two subsets: medication, relation
    # remove the answers whose lengths are larger than max_answer_length (token)
    subsets = []
    for subset in dataset_json['data']:
        data = []
        for para in tqdm(subset['paragraphs']):
            subdata = {"title": '', 'paragraphs': []}
            new_para = {'context': para['context'], 'qas': []}
            for qa in para['qas']:
                new_answers = []
                for answer in qa['answers']:
                    answer_length = len(answer['text'].split())
                    if answer_length < max_answer_length:
                        new_answers.append(answer)
                if len(new_answers) > 0:
                    new_qa = {'question': qa['question'], 'id': qa['id'], 'answers': new_answers}
                    new_para['qas'].append(new_qa)

            subdata['paragraphs'].append(new_para)
            subdata['title'] = ' '.join(new_para['context'].split()[:2])
            data.append(subdata)
        subset_json = {'data': data, 'version': 1.0}
        # print("saving...")
        # json.dump(subset_json, open(os.path.join(output_dir,subset['title'] + '.json'), 'w'))
        subsets.append(subset_json)
    return subsets


def show_dataset_statistics(data_json, dataset_name):
    print('#' * 50)
    print("Showing the statistics of :", dataset_name)
    note_num = len(data_json['data'])
    question_num = 0
    qa_pair_num = 0
    para_num = 0
    question_len = 0
    answers_len = 0
    context_len = 0
    answers_num = 0
    for paras in data_json['data']:
        for para in paras['paragraphs']:
            para_num += 1
            context_len += len(para['context'].split(" "))
            for qa in para['qas']:
                question_num += 1
                qa_pair_num += len(qa['answers'])
                question_len += len(qa['question'].split(" "))
                for ans in qa['answers']:
                    answers_num += 1
                    answers_len += len(ans['text'].split(" "))

    print("Avg answer length: %.2f " % (answers_len / answers_num))
    print("Avg question length: %.2f" % (question_len / question_num))
    print("Avg context length: %.2f" % (context_len / para_num))

    print('Note (context): {}, Questions: {}, QA_pair: {}'.format(note_num, question_num, qa_pair_num))


def split_train_dev_test_per_note(data_json, output_dir, train_ratio=0.7, dev_ratio=0.1, subset_name='relation'):
    assert train_ratio + dev_ratio <= 1.0

    note_num = len(data_json['data'])
    train_note_num = int(train_ratio * note_num)
    dev_note_num = int(dev_ratio * note_num)
    note_list = random.sample(range(note_num), note_num)

    train_data = {'data': [], 'version': 1.0}

    for i in range(train_note_num):
        note = data_json['data'][note_list[i]]
        train_data['data'].append(note)

    json.dump(train_data, open(os.path.join(output_dir, '{}-train.json'.format(subset_name)), 'w'))

    dev_data = {'data': [], 'version': 1.0}
    for i in range(train_note_num, train_note_num + dev_note_num):
        note = data_json['data'][note_list[i]]
        dev_data['data'].append(note)
    json.dump(dev_data, open(os.path.join(output_dir, '{}-dev.json'.format(subset_name)), 'w'))

    test_data = {'data': [], 'version': 1.0}
    for i in range(train_note_num + dev_note_num, note_num):
        note = data_json['data'][note_list[i]]
        test_data['data'].append(note)
    json.dump(test_data, open(os.path.join(output_dir, '{}-test.json'.format(subset_name)), 'w'))


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
