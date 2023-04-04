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


def get_test_sentences(test_file):
    sentences = []
    sentence = []

    num_words = 0 # to get the number of words in the entire test file

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
            if word.lower() not in observation_probabilities[curr_tag_i].keys():
                observation_probabilities[curr_tag_i][word.lower()] = 1
            else:
                observation_probabilities[curr_tag_i][word.lower()] += 1


            if i < len(sentence) - 1:
                next_tag = sentence[i + 1][1]
                next_tag_i = TAGS.index(next_tag)

                transition_probabilities[curr_tag_i][next_tag_i] += 1



    # print("here are the transition counts[0] before normalizing:")
    # print(transition_probabilities[0])
    # print(" ")

    # Normalize the initital probabilities
    initial_probabilities /= num_sentences  

    # have this smoothing constant to avoid dividing by zero in the transition probability normalization
    smoothing_constant = 1e-10

    # Normalize the transition probabilities - each row has P(next tag | curr tag) - normalize by dividing each by tag_counts(curr_tag)
    for i in range(len(transition_probabilities)):
        transition_probabilities[i] = (transition_probabilities[i] + smoothing_constant)/ tag_counts[i]

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

    return initial_probabilities, transition_probabilities, observation_probabilities




def viterbi_algorithm_V2(sentence, initial_probabilities, transition_probabilities, observation_probabilities):
    
    n_states = len(initial_probabilities)
    n_observations = len(sentence)

    prob = np.zeros((n_observations, n_states))
    prev = np.zeros((n_observations, n_states), dtype=int)

    # Determine t=0 valuess
    for i in range(n_states):
        if sentence[0].lower() in observation_probabilities[i]:
            prob[0, i] = initial_probabilities[i] * observation_probabilities[i][sentence[0].lower()]
        
        elif re.search(r'[,.!?]', sentence[0]):  # Punctuation should always be tagged as 'PUN'
            if TAGS[i] == 'PUN':
                prob[0, i] = 1.0 # definitely PUN so p=1 here
        
        prev[0, i] = -1

    # For t=1 to t=length(sentence)-1, find each curr state's most likely prev state
    for t in range(1, n_observations):
        for i in range(n_states):
            
            # make sure .lower() since observation_prob has the same
            if sentence[t].lower() in observation_probabilities[i]:
                obs_prob = observation_probabilities[i][sentence[t].lower()]
            
            elif re.search(r'[,.!?]', sentence[t]):  # Punctuation should always be tagged as 'PUN'
                if TAGS[i] == 'PUN':
                    obs_prob = 1.0
                else:
                    obs_prob = 0.0
            else:
                obs_prob = 0.0

            temp_probs = prob[t - 1] * transition_probabilities[:, i] * obs_prob
            x = np.argmax(temp_probs)
            prob[t, i] = temp_probs[x]
            prev[t, i] = x

    # Backtrack to get most likely sequence of states
    best_path = [np.argmax(prob[-1])]
    for t in range(n_observations - 1, 0, -1):
        best_path.append(prev[t, best_path[-1]])

    best_path.reverse()
    best_path = [TAGS[i] for i in best_path]

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
    testing_sentences = get_test_sentences(args.testfile)

    # get our probability tables - build HMM
    initial_pr, transition_pr, observation_pr = generate_tables_V2(training_sentences)


    tagged_sentences = []

    for sentence in testing_sentences:
        tagged_sentence = viterbi_algorithm_V2(sentence, initial_pr, transition_pr, observation_pr)
        tagged_sentences.append(tagged_sentence)

    print(tagged_sentences)

    # now put the results into the output file
    with open(args.outputfile, 'w') as output_file:
        for tagged_sentence in tagged_sentences:
            for word, tag in tagged_sentence:
                output_file.write(f"{word} : {tag}\n")

