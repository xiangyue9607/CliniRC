import re
import pprint
import json
import pandas as pd
import random
import numpy as np
import glob
import re
import os
import copy
import ast
import uuid

with open('../src/ClarityNLP/nlp/algorithms/sec_tag/data/concepts_and_synonyms.txt', "r") as f:
    sec_tag_labels = f.readlines()

sec_tag_labels = sec_tag_labels[1:]
headers = set([line.strip().split("\t")[-2].lower().strip().replace('_', ' ') for line in sec_tag_labels])


def split_sections(lines):
    """
    Header Cases

    In order to split the clinical notes into sections, we notice that most sections begin with easily identifiable headers.
    To detect these headers we use a combination of heuristics such as whether the line contains colons, all uppercase formatting or
    phrases found in a list of clinical headers taken from SecTag {Denny et al. 2008}.

    1)
    Match the header regex AND
        (Upper AND < 4) OR Header Name OR Nothing after ':'

    2)
    Word segment in line (matches alpha) AND
    First word segment is short AND
    More than one line in section AND
    Last line ends in period AND
        (Header Name OR Upper)
    """

    # Regex (Non Letters) (Letter without ':') (Post Colon)
    header_regex = re.compile("([^A-Za-z]*)([^:]+):(.*)")

    # Words in first part of header. '2. Pulmonary disease. The patient ...' group(1) = Pulmonary disease
    alpha = re.compile("[^A-Za-z]*([A-Za-z ]*)[^A-Za-z](.*)")

    sections = []
    section_nums = []
    section = []
    lines_in_section = []

    for line_num, original_line in enumerate(lines):
        line = original_line.strip()

        is_header = False

        match = header_regex.match(line)
        alpha_match = alpha.match(line)

        # If there's a match check first group
        if match:
            header = match.group(2)
            post_colon = match.group(3).strip()

            upper = header.isupper()
            short = len(header.split(" ")) < 5

            ends_in_colon = post_colon == ""

            # All caps is always header
            if (upper and short) or ends_in_colon:
                is_header = True
            # If header in headers
            else:
                header = header.strip().lower()
                is_header = header in headers

        # If no match check first word section of whole line as header
        elif alpha_match and len(section) > 1:
            last_line = section[-1]

            if last_line != "" and last_line[-1] == ".":
                header = alpha_match.group(1).strip()

                if len(header.split(" ")) < 5:
                    upper = header.isupper()
                    in_headers = header.lower() in headers

                    if upper or in_headers:
                        is_header = True

        if is_header and section != []:
            sections.append(section)
            section_nums.append(lines_in_section)

            section = []
            lines_in_section = []

        section.append(original_line)
        lines_in_section.append(line_num)

    sections.append(section)
    section_nums.append(lines_in_section)

    return sections, section_nums

# %%

dataset_names = ["../data/data*"]
datasets = {}

for name in dataset_names:
    print(name)

    qa_pairs = {}
    mimic_docs = 0
    contexts = []
    qa_list = []

    files = glob.glob(name)

    for file in files:
        with open(file, "r") as f:
            json_file = f.read()

            result = json.loads(json_file)

        datasets[file] = result

f = datasets['../data/data.json']


# %%

def flip_section_list(section_nums):
    line_to_section = {}

    for section_num, line_list in enumerate(section_nums):

        for line in line_list:
            line_to_section[line] = section_num

    return line_to_section


# %%

errors = 0
orig_num_answers = 0
new_num_answers = 0

emrqa_datasets = {}

for task in f['data']:

    title = task['title']

    if title in ['medication', 'relations']:

        reports = task['paragraphs']
        documents = []

        for report in reports:

            note_id = report['note_id']
            text_lines = report['context']
            qas = report['qas']

            new_report = {"title": note_id}
            new_paragraphs = []

            sections, section_nums = split_sections(text_lines)
            line_to_section = flip_section_list(section_nums)

            section_num_to_qas = {}

            for qa in qas:
                answers = qa['answers']

                # Group answers by section number for one qa
                section_num_to_answers = {}

                for answer in answers:
                    orig_num_answers += 1

                    evidence_lines = answer['evidence_start']
                    answer_starts = answer['answer_start']

                    answer_texts = answer['text']
                    evidences = answer['evidence']

                    if isinstance(evidence_lines, int):
                        evidence_lines = [evidence_lines]
                        answer_starts = [answer_starts]
                        answer_texts = [answer_texts]
                        evidences = [evidences]

                    for evidence_line, evidence, answer_text in zip(evidence_lines, evidences, answer_texts):
                        new_answer = copy.deepcopy(answer)
                        new_num_answers += 1

                        section_num = line_to_section[evidence_line - 1]

                        first_section_line = section_nums[section_num][0]
                        new_evidence_line = evidence_line - first_section_line

                        new_answer['evidence_start'] = new_evidence_line
                        new_answer['answer_start'][0] = new_evidence_line
                        new_answer['evidence'] = evidence
                        new_answer['text'] = answer_text
                        new_answer['answer_entity_type'] = 'single'

                        section = sections[section_num]

                        if section_num in section_num_to_answers:
                            new_answers = section_num_to_answers[section_num]
                        else:
                            new_answers = []

                        section_text = ''.join(section)

                        evidence_in_section = section[new_evidence_line - 1]

                        if evidence in section_text:
                            new_answers.append(new_answer)
                            section_num_to_answers[section_num] = new_answers
                        else:
                            errors += 1

                # Add qas with new set of answers for each section num
                for section_num in section_num_to_answers.keys():
                    new_answers = section_num_to_answers[section_num]

                    new_qa = copy.deepcopy(qa)
                    new_qa['answers'] = new_answers

                    if section_num in section_num_to_qas:
                        new_qas = section_num_to_qas[section_num]
                    else:
                        new_qas = []

                    new_qas.append(new_qa)
                    section_num_to_qas[section_num] = new_qas

            for section_num in section_num_to_qas.keys():
                section = sections[section_num]

                paragraph = {"note_id": note_id + "_" + str(section_num),
                             "context": section,
                             "qas": section_num_to_qas[section_num]}
                new_paragraphs.append(paragraph)

            new_report['paragraphs'] = new_paragraphs

            documents.append(new_report)

        print("Saving {}".format(title))
        emrqa_datasets[title] = documents

# %%

temp_emrqa = copy.deepcopy(emrqa_datasets)

# %%

emrqa_datasets = copy.deepcopy(temp_emrqa)

# %%

errors, orig_num_answers, new_num_answers

# %%

len(emrqa_datasets['medication']), len(emrqa_datasets['relations'])


# %% md

## Making SQUAD JSON and Cleaning QA's

# %%

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


def locate_answer_start(evidence, context):
    while evidence[-1] in [',', '.', '?', '!', '-', ' ']:
        evidence = evidence[:-1]
    char_pos = -1
    temp_evidence = evidence
    final_evidence = temp_evidence
    while char_pos == -1:
        char_pos = context.find(temp_evidence)
        final_evidence = temp_evidence
        temp_evidence = ' '.join(temp_evidence.split()[:-1])
    return char_pos, final_evidence


def combine_answer_lines(answers):
    line_numbers = []
    answers_by_line = {}

    combined_answers = []

    for answer in answers:
        line = answer['evidence_start']

        line_numbers.append(line)
        answers_by_line[line] = answer

    ordered_line_numbers = np.sort(line_numbers)

    prev_line = ordered_line_numbers[0]
    combined_line = answers_by_line[prev_line]['evidence']

    for line_num in ordered_line_numbers[1:]:
        if line_num - prev_line == 1:
            combined_line += " " + answers_by_line[line_num]['evidence']
        else:
            combined_answers.append({'evidence': combined_line})
            combined_line = answers_by_line[line_num]['evidence']

        prev_line = line_num

    combined_answers.append({'evidence': combined_line})

    return combined_answers


# %%

answers_checked = 0
longer_than = 0

long_answers = 0

new_emrqa = {}

for title in emrqa_datasets.keys():

    records = emrqa_datasets[title]
    new_records = []

    for record in records:

        record_id = record['title']
        sections = record['paragraphs']

        new_record = {'title': record_id}

        new_sections = []
        context_set = set()

        for section in sections:
            new_section = {}

            text_list = section['context']

            context = " ".join((" ".join(text_list).split()))
            context = prune_sentence(context)

            qas = section['qas']
            new_qas = []

            for qa in qas:
                questions = qa['question']
                answers = qa['answers']

                new_answers = []

                if len(answers) > 1:
                    answers = combine_answer_lines(answers)

                # Clean Answers and Check that they are in the context
                for answer in answers:
                    evidence = answer['evidence']
                    evidence = prune_sentence(evidence)

                    evidence_length = len(evidence.split())

                    if len(evidence.strip()) > 0 and evidence_length < 20:
                        answer_start, evidence = locate_answer_start(evidence, context)

                        new_answers.append({'answer_start': answer_start, 'text': evidence})

                        assert evidence in context

                        answers_checked += 1
                    else:
                        long_answers += 1

                # Add new qa pair for each question paraphrase
                for question in questions:

                    # If answers were too long don't append question
                    if len(new_answers) > 0:
                        new_qas.append({'question': question, 'id': str(uuid.uuid1().hex), 'answers': new_answers})

            # If all questions in section had longer than acceptable answers
            if len(new_qas) > 0:
                context_set.add(section['note_id'])
                new_sections.append({'qas': new_qas, 'context': context})
            else:
                print("Lost Section")

        assert len(context_set) == len(new_sections)

        # if all sections in record only had longer than accetable answers
        if len(new_sections) > 0:
            new_record['paragraphs'] = new_sections
            new_records.append(new_record)
        else:
            print("Lost Whole record")

    new_emrqa[title] = new_records

print(long_answers)

# %%

num_qas = {}
num_contexts = {}

for title in new_emrqa.keys():

    num_qa = 0
    num_context = 0

    records = new_emrqa[title]

    for record in records:
        sections = record['paragraphs']
        num_context += 1
        for section in sections:
            qas = section['qas']

            for qa in qas:
                num_qa += 1

    num_qas[title] = num_qa
    num_contexts[title] = num_context

print(num_qas)
print(num_contexts)


# %%

def add_header_and_save(emrqa_datasets):
    for name in emrqa_datasets.keys():
        emrqa_datasets[name] = {'version': '1', 'data': emrqa_datasets[name]}

        json.dump(emrqa_datasets[name], open('../data/{}.json'.format(name), 'w'))

    return emrqa_datasets


# %%

new_emrqa = add_header_and_save(new_emrqa)
new_emrqa['relations']['data'][0]