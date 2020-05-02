import json
import numpy as np
import glob
import re
import copy
import uuid
from preprocessing import prune_sentence

with open('data/concepts_and_synonyms.txt', "r") as f:
    sec_tag_labels = f.readlines()

sec_tag_labels = sec_tag_labels[1:]
headers = set([line.strip().split("\t")[-2].lower().strip().replace('_', ' ') for line in sec_tag_labels])

def split_sections(lines):
    """

    :param lines: Clinical note in the form of a list of strings. Each element is a line.
    :return:
        - sections: List of strings, each element is a section.
        - section_nums: List of integer lists, each element is a list of lines belonging to each section.

    In order to split the clinical notes into sections, we notice that most sections begin with easily identifiable headers.
    To detect these headers we use a combination of heuristics such as whether the line contains colons, all uppercase formatting or
    phrases found in a list of clinical headers taken from SecTag {Denny et al. 2008}.

    Conditions for Header Detection
    1) (Line matches with header regex) AND
       (Group 1 of regex is (Upper AND shorter than 4) OR in the header list OR there is nothing in this line following ':')

    2) (Line matches alpha regex) AND
       (first word segment is short) AND
       (more than one line in section) AND
       (last line ends in period) AND
       (Group 1 is (in header OR Upper)
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

        #Add previous section if it exists and we encounter a header
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

def load_emrqa_datasets(data_dir="../data/datasets/*json"):
    """

    :return: dictionary from filename to decoded json objects in data directory
    """

    datasets = {}

    files = glob.glob(data_dir)

    for file in files:
        with open(file, "r") as f:
            json_file = f.read()

            result = json.loads(json_file)

        datasets[file] = result

    return datasets

def flip_section_list(section_nums):
    line_to_section = {}

    for section_num, line_list in enumerate(section_nums):

        for line in line_list:
            line_to_section[line] = section_num

    return line_to_section


def group_answers_by_section(qa, sections, section_nums, line_to_section, orig_num_answers, new_num_answers, errors):
    new_answers = []
    answers = qa['answers']

    # Group answers by section number for one qa
    section_num_to_answers = {}

    for answer in answers:
        orig_num_answers += 1

        evidence_lines = answer['evidence_start']
        answer_texts = answer['text']
        evidences = answer['evidence']

        if isinstance(evidence_lines, int):
            evidence_lines = [evidence_lines]
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

            if evidence in section_text:
                new_answers.append(new_answer)
                section_num_to_answers[section_num] = new_answers
            else:
                errors += 1

    return section_num_to_answers, orig_num_answers, new_num_answers, errors

def create_split_docs_emrqa(emrqa):
    """

    :param emrqa: Decoded emrQA dataset json
    :return: json dataset with the same structure as the emrQA but linking each question with a section in a document
    instead of the whole report.

    """

    errors = 0
    orig_num_answers = 0
    new_num_answers = 0

    emrqa_datasets = {}

    for task in emrqa['data']:

        title = task['title']

        #Splitting only medication and relations datasets due for evaluation
        if title in ['medication', 'relations']:

            reports = task['paragraphs']
            documents = []

            #Looping through all medical reports
            for report in reports:

                note_id = report['note_id']
                text_lines = report['context']
                qas = report['qas']

                new_report = {"title": note_id}
                new_paragraphs = []

                #Splitting Sections
                sections, section_nums = split_sections(text_lines)

                #Reversing the map from lines to section numbers
                line_to_section = flip_section_list(section_nums)


                section_num_to_qas = {}

                #Looping through all questions for this report, each question might have multiple answers
                for qa in qas:

                    section_num_to_answers, orig_num_answers, new_num_answers, errors  = group_answers_by_section(qa, sections, section_nums, line_to_section,
                                                                      orig_num_answers, new_num_answers, errors)

                    # Aggregate qas with equivalent section num
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

    return emrqa_datasets, orig_num_answers, new_num_answers, errors


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

def transform_emrqa_to_squad_format(emrqa_format_dataset):

    answers_checked = 0
    long_answers = 0

    squad_format_dataset = {}

    for title in emrqa_format_dataset.keys():

        records = emrqa_format_dataset[title]
        new_records = []

        for record in records:

            record_id = record['title']
            sections = record['paragraphs']

            new_record = {'title': record_id}

            new_sections = []
            context_set = set()

            for section in sections:

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

                        # If all answers were too long don't append question
                        if len(new_answers) > 0:
                            new_qas.append({'question': question, 'id': str(uuid.uuid1().hex), 'answers': new_answers})

                # If all questions in section had longer than acceptable answers
                if len(new_qas) > 0:
                    context_set.add(section['note_id'])
                    new_sections.append({'qas': new_qas, 'context': context})
                # else:
                #     print("Lost Section")

            assert len(context_set) == len(new_sections)

            # if all sections in record only had longer than accetable answers
            if len(new_sections) > 0:
                new_record['paragraphs'] = new_sections
                new_records.append(new_record)
            # else:
            #     print("Lost Whole record")

        squad_format_dataset[title] = new_records

    return squad_format_dataset, answers_checked, long_answers

def count_squad_format_qas_and_contexts(noheader_squad_dataset):

    num_qas = {}
    num_contexts = {}

    for title in noheader_squad_dataset.keys():

        num_qa = 0
        num_context = 0

        records = noheader_squad_dataset[title]

        for record in records:
            sections = record['paragraphs']
            num_context += 1
            for section in sections:
                qas = section['qas']

                for qa in qas:
                    num_qa += 1

        num_qas[title] = num_qa
        num_contexts[title] = num_context

    return num_qas, num_contexts

def add_header_and_save(emrqa_datasets, directory):
    for name in emrqa_datasets.keys():
        emrqa_datasets[name] = {'version': '1', 'data': emrqa_datasets[name]}

        json.dump(emrqa_datasets[name], open(directory + '{}.json'.format(name), 'w'))

    return emrqa_datasets