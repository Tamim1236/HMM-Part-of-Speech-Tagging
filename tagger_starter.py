import os
import sys
import argparse

import re
import numpy as np

TAGS = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
        "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
        "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
        "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
        'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
        'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
        'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
        'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
        'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']



# !note: what is we have the end of a speech, like "Can you tell me how much this costs?" - here the end of the sentence should be the  "  at the end 
def get_sentences(training_list):
    
    sentence = [] # keep track of current sentence
    sentences = [] # add all sentences to this list
    
    
    for file in training_list:
        with open(file, 'r') as f:
            for line in f:
                #print(line)
                # .strip() removes leading and trailing characters (removes '\n')
                # .split() with index 1 makes sure we don't mess up when there are multiple colons in a line, like ": : PUN"
                word, tag = line.strip().split(' : ', 1)
                sentence.append((word, tag))
                
                # check whether the word is an ending punctuation
                if re.search(r'[.!?]', word):
                    sentences.append(sentence)
                    sentence = [] #reset current sentence
    
    #print(f"here is how many sentences we have in the text: {len(sentences)}")

    return sentences

# def get_test_sentences(test_file):
#     sentences = []
#     sentence = []

#     with open(test_file, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line or re.search(r'[.!?]', line):
#                 if sentence:
#                     sentences.append(sentence)
#                     sentence = []
#             else:
#                 sentence.append(line)

#     return sentences

def get_test_sentences(test_file):
    sentences = []
    sentence = []

    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            words = line.split()
            for word in words:
                sentence.append(word)
                if re.search(r'[.!?]', word):
                    sentences.append(sentence)
                    sentence = []

    return sentences


# def generate_tables(sentences):
#     initial_probabilities = {}
#     num_sentences = len(sentences)

#     transition_probabilities = {}
#     observation_probabilities = {}
#     tag_counts = {}

#     for sentence in sentences:

#         # procedure for initial_probabilities
#         initial_tag = sentence[0][1]
#         if initial_tag in initial_probabilities:
#             initial_probabilities[initial_tag] += 1
#         else:
#             initial_probabilities[initial_tag] = 1

#         for i in range(len(sentence)):
#             word, tag = sentence[i]

#             if tag in tag_counts:
#                 tag_counts[tag] += 1
#             else:
#                 tag_counts[tag] = 1

#             # Update observation_probabilities
#             if tag not in observation_probabilities:
#                 observation_probabilities[tag] = {}
#             observation_probabilities[tag][word] = observation_probabilities[tag].get(word, 0) + 1

#             if i < len(sentence) - 1:
#                 next_tag = sentence[i + 1][1]

#                 # Update transition_probabilities
#                 if tag not in transition_probabilities:
#                     transition_probabilities[tag] = {}
#                 transition_probabilities[tag][next_tag] = transition_probabilities[tag].get(next_tag, 0) + 1

#     # now normalize initial probability values using num_sentences
#     for tag in initial_probabilities:
#         initial_probabilities[tag] /= num_sentences

#     # Update transition probabilities
#     for current_tag in transition_probabilities:
#         total_transitions = sum(transition_probabilities[current_tag].values())
        
#         for next_tag in transition_probabilities[current_tag]:
#             transition_probabilities[current_tag][next_tag] /= total_transitions

#     # Update observation probabilities
#     for tag in observation_probabilities:
#         total_observations = sum(observation_probabilities[tag].values())
        
#         for word in observation_probabilities[tag]:
#             observation_probabilities[tag][word] /= total_observations

#     # print("here are the initial probs:")
#     # print(initial_probabilities)
#     # print(f"All the probabilities in initial_probabilities sum up to {sum(initial_probabilities.values())}")

#     # print("here are the transition probs:")
#     # print(transition_probabilities)

#     # print("here are the observation probs:")
#     # print(observation_probabilities)

#     return initial_probabilities, transition_probabilities, observation_probabilities



def generate_tables_V2(sentences):

    num_sentences = 0
    
    initial_probabilities = np.zeros(91, dtype=float) # 1x91 matrix
    transition_probabilities = np.zeros((91,91), dtype=float) # 91x91 matrix
    observation_probabilities = [{} for i in range(91)] # creating an empty list where each row is an empty dictionary (for the words)

    #print(observation_probabilities)


    tag_counts = np.zeros(91, dtype=int) # 1x91 matrix to keep track of how many times tags appeared across all sentences

    
    # print(initial_probabilities)
    # print(f"here is the size of initial_probabilities: {initial_probabilities.shape}")
    # print()
    # print()
    # print(transition_probabilities)
    # print(f"here is the size of transition_probabilities: {transition_probabilities.shape}")

    #print(sentences)

    for sentence in sentences:
        num_sentences += 1

        # procedure for initial_probabilities
        initial_tag = sentence[0][1]
        initial_tag_i = TAGS.index(initial_tag)
        #print(initial_tag_i)

        initial_probabilities[initial_tag_i] += 1 # increment (initial) count for this tag

        # go through entire sentence
        for i in range(len(sentence)):
            
            word, tag = sentence[i]
            curr_tag_i = TAGS.index(tag)
            tag_counts[curr_tag_i] += 1 # increment count for this tag - increments up to and including last tag in sentence

            # either add new key-value pair (word:count) or increment count
            # make sure to add lower-case for word
            if word not in observation_probabilities[curr_tag_i].keys():
                observation_probabilities[curr_tag_i][word] = 1
            else:
                observation_probabilities[curr_tag_i][word] += 1


            if i < len(sentence) - 1:
                next_tag = sentence[i + 1][1]
                next_tag_i = TAGS.index(next_tag)

                transition_probabilities[curr_tag_i][next_tag_i] += 1



    # print("here are the transition counts[0] before normalizing:")
    # print(transition_probabilities[0])
    # print(" ")

    # Normalize the initital probabilities
    initial_probabilities /= num_sentences  

    # Normalize the transition probabilities - each row has P(next tag | curr tag) - normalize by dividing each by tag_counts(curr_tag)
    for i in range(len(transition_probabilities)):
        transition_probabilities[i] /= tag_counts[i]

    # Normalize the observation probabilities
    for i in range(len(observation_probabilities)):
        for key, value in observation_probabilities[i].items():
            observation_probabilities[i][key] = value/tag_counts[i]

    
    # print(initial_probabilities)
    # print(f"here is the new size of initial_probabilities: {initial_probabilities.shape}")
    # print(f"There were this many sentences: {num_sentences}")

    # print("here are the tag counts:")
    # print(tag_counts)
    # print(" ")

    # print("here are the transition counts (some rows):")
    # print()
    # print(transition_probabilities[0])
    # print(transition_probabilities[20])
    
    print("here are the observational probabilities:")
    print(observation_probabilities)





def viterbi_algorithm(sentence, initial_probabilities, transition_probabilities, observation_probabilities):
    viterbi = []
    backpointer = []

    first_word = sentence[0]
    first_viterbi = {}
    first_backpointer = {}

    # for tag in initial_probabilities:
    #     if tag in observation_probabilities and first_word in observation_probabilities[tag]:
    #         first_viterbi[tag] = initial_probabilities[tag] * observation_probabilities[tag][first_word]
    #     else:
    #         first_viterbi[tag] = 0
    #     first_backpointer[tag] = None

    max_prob = 0
    best_starting_tag = None

    for tag in initial_probabilities:
        first_viterbi[tag] = 0
        first_backpointer[tag] = None
        if tag in observation_probabilities and first_word in observation_probabilities[tag]:
            prob = initial_probabilities[tag] * observation_probabilities[tag][first_word]
            if prob > max_prob:
                max_prob = prob
                best_starting_tag = tag

    first_viterbi[best_starting_tag] = max_prob
    first_backpointer[best_starting_tag] = None



    viterbi.append(first_viterbi)
    backpointer.append(first_backpointer)

    for word_index in range(1, len(sentence)):
        current_viterbi = {}
        current_backpointer = {}
        prev_viterbi = viterbi[-1]

        for current_tag in initial_probabilities:
            max_prob = 0
            best_previous_tag = None

            for previous_tag in initial_probabilities:
                if previous_tag in transition_probabilities and current_tag in transition_probabilities[previous_tag]:
                    transition_prob = transition_probabilities[previous_tag][current_tag]
                else:
                    transition_prob = 0

                if current_tag in observation_probabilities and sentence[word_index] in observation_probabilities[current_tag]:
                    observation_prob = observation_probabilities[current_tag][sentence[word_index]]
                else:
                    observation_prob = 0

                prob = prev_viterbi[previous_tag] * transition_prob * observation_prob

                if prob > max_prob:
                    max_prob = prob
                    best_previous_tag = previous_tag

            current_viterbi[current_tag] = max_prob
            current_backpointer[current_tag] = best_previous_tag

        viterbi.append(current_viterbi)
        backpointer.append(current_backpointer)

    best_final_tag = max(viterbi[-1], key=viterbi[-1].get)
    best_path = [best_final_tag]

    for bp in reversed(backpointer[1:]):
        best_previous_tag = bp[best_final_tag]
        if best_previous_tag:
            best_path.insert(0, best_previous_tag)
            best_final_tag = best_previous_tag

    return list(zip(sentence, best_path))







if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))


    print("Starting the tagging process.")

    print("here is the training list:")
    #print(training_list)

    training_sentences = get_sentences(training_list)
    # initial_pr, transitional_pr, observational_pr = generate_tables(training_sentences)

    testing_sentences = get_test_sentences(args.testfile)

    #print(testing_sentences)


    # tagged_sentences = []
    # for sentence in testing_sentences:
    #     tagged_sentence = viterbi_algorithm(sentence, initial_pr, transitional_pr, observational_pr)
    #     tagged_sentences.append(tagged_sentence)

    
    #print(tagged_sentences)



    #print(sentences)

    # generate_tables(training_list)

    generate_tables_V2(training_sentences)


