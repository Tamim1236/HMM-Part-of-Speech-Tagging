import os
import sys
import argparse
from collections import defaultdict, Counter

def read_training_data(training_files):
    data = []

    for file in training_files:
        with open(file, 'r') as f:
            for line in f:
                if line.strip():
                    word, tag = line.strip().split(' : ')
                    data.append((word, tag))

    return data

def count_occurrences(data):
    tag_count = Counter()
    tag_transition_count = defaultdict(Counter)
    word_tag_count = defaultdict(Counter)

    previous_tag = None
    for word, tag in data:
        tag_count[tag] += 1
        word_tag_count[word][tag] += 1

        if previous_tag:
            tag_transition_count[previous_tag][tag] += 1

        previous_tag = tag

    return tag_count, tag_transition_count, word_tag_count

def calculate_probabilities(tag_count, tag_transition_count, word_tag_count):
    initial_probabilities = {tag: count / sum(tag_count.values()) for tag, count in tag_count.items()}
    transition_probabilities = {tag1: {tag2: count / sum(tag_transition_count[tag1].values()) for tag2, count in counts.items()} for tag1, counts in tag_transition_count.items()}
    observation_probabilities = {word: {tag: count / sum(word_tag_count[word].values()) for tag, count in counts.items()} for word, counts in word_tag_count.items()}

    return initial_probabilities, transition_probabilities, observation_probabilities

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--trainingfiles",
#         action="append",
#         nargs="+",
#         required=True,
#         help="The training files."
#     )
#     parser.add_argument(
#         "--testfile",
#         type=str,
#         required=True,
#         help="One test file."
#     )
#     parser.add_argument(
#         "--outputfile",
#         type=str,
#         required=True,
#         help="The output file."
#     )
#     args = parser.parse_args()

#     training_list = args.trainingfiles[0]
#     print("training files are {}".format(training_list))

#     print("test file is {}".format(args.testfile))

#     print("output file is {}".format(args.outputfile))

#     print("Starting the tagging process.")
    
#     training_data = read_training_data(training_list)
#     tag_count, tag_transition_count, word_tag_count = count_occurrences(training_data)
#     initial_probabilities, transition_probabilities, observation_probabilities = calculate_probabilities(tag_count, tag_transition_count, word_tag_count)

#     print(initial_probabilities)

training_data = read_training_data("training1.txt")
tag_count, tag_transition_count, word_tag_count = count_occurrences(training_data)
initial_probabilities, transition_probabilities, observation_probabilities = calculate_probabilities(tag_count, tag_transition_count, word_tag_count)

print(initial_probabilities)
