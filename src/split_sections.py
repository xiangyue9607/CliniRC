from section_split_utils import *

#Change to emrQA json filename
emrqa_dir = 'data/datasets/'
emrqa_filename = 'data.json'

#Loading emrQA datasets from data directory
datasets = load_emrqa_datasets(emrqa_dir+emrqa_filename)
emrqa_json = datasets[emrqa_dir + emrqa_filename]

#Splitting emrQA questions by clinical record sections
emrqa_datasets, orig_num_answers, new_num_answers, errors = create_split_docs_emrqa(emrqa_json)

print("Number of errors from extracting correct sections for each answer: {}".format(errors))
print("Number of answers in original dataset: {}".format(orig_num_answers))
print("Number of answers in new dataset (Should be more): {}".format(new_num_answers))

#Transforming to Squad format, preprocessing the context/answers and filtering long questions
headerless_squad_emrqa, answers_checked, long_answers = transform_emrqa_to_squad_format(emrqa_datasets)

print("Number of removed answers due to length: {}".format(long_answers))

#Verifying QA Pair Counts
num_qas, num_contexts = count_squad_format_qas_and_contexts(headerless_squad_emrqa)

print("Number of QA pairs in new SQUAD format dataset: {}".format(num_qas))
print("Number of contexts in new SQUAD format dataset: {}".format(num_contexts))

#Adding Header to each sub emrQA dataset and create medication.json and relations.json
new_emrqa = add_header_and_save(headerless_squad_emrqa, emrqa_dir)