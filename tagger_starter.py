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
                if re.search(r'[!?.]', word):
                    sentences.append(sentence)
                    sentence = [] #reset current sentence
    
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
                if re.search(r'[.?!]', word):
                    sentences.append(sentence)
                    sentence = []

    return sentences


def generate_tables_V2(sentences):

    num_sentences = 0
    
    initial_probabilities = np.zeros(91, dtype=float) # 1x91 matrix
    transition_probabilities = np.zeros((91,91), dtype=float) # 91x91 matrix
    observation_probabilities = [{} for i in range(91)] # creating an empty list where each row is an empty dictionary (for the words)

    tag_counts = np.zeros(91, dtype=int) # 1x91 matrix to keep track of how many times tags appeared across all sentences

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

    # have this smoothing constant to avoid dividing by zero in the transition probability normalization
    # this way if we make a wrong prediction, say AV0, and then we are trying to get probability of AV0
    # from initial, or probability of AV0->X from transition, even if those were initially zero, we can account for it
    smoothing_constant = 1e-10

    # Normalize the initital probabilities
    #initial_probabilities /= num_sentences  
    initial_probabilities = (initial_probabilities + smoothing_constant) / (num_sentences + smoothing_constant * len(TAGS))


    # Normalize the transition probabilities - each row has P(next tag | curr tag) - normalize by dividing each by tag_counts(curr_tag)
    for i in range(len(transition_probabilities)):
        #transition_probabilities[i] = (transition_probabilities[i] + smoothing_constant)/ tag_counts[i]
        transition_probabilities[i] = (transition_probabilities[i] + smoothing_constant) / (tag_counts[i] + smoothing_constant * len(TAGS))

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

    return initial_probabilities, transition_probabilities, observation_probabilities, tag_counts


def viterbi_algorithm_V2(sentence, initial_probabilities, transition_probabilities, observation_probabilities, tag_counts):
    
    num_states = len(initial_probabilities)
    num_observations = len(sentence)

    # initialize the prob and prev matrices
    prob = np.zeros((num_observations, num_states)) # size = num words x 91
    prev = np.zeros((num_observations, num_states), dtype=int) # size = num words x 91

    # Go through all tags - Get values at time t=0 for first word
    first_word = sentence[0].lower()
    for i in range(num_states):
        if first_word in observation_probabilities[i]:
            prob[0, i] = initial_probabilities[i] * observation_probabilities[i][first_word]
        
        # Specifically for the case when we have punctuation => always predict 'PUN'
        # so this is if the first word wasnt observed, but it is punctuation => we are certain it is PUN
        elif re.search(r'[,.!?]', first_word):
            # if its punctuation, then the prob for PUN=1 (guaranteed) and rest remain zero (initialized in np.zeros())
            if TAGS[i] == 'PUN':
                prob[0, i] = 1.0
        
        # !!!what if this first word is one we haven't seen before? Shouldnt just be leaving it as zero
        
        # no previous since this is the first tag
        prev[0, i] = -1

    # Recursive step from t=1 to t=len(sentence)-1
    for t in range(1, num_observations):
        for i in range(num_states):
            
            curr_word = sentence[t].lower()

            if curr_word in observation_probabilities[i]:
                observation_prob = observation_probabilities[i][curr_word]
            
            # Specifically for the case when we have punctuation => always predict 'PUN'
            elif re.search(r'[,.!?]', curr_word):
                if TAGS[i] == 'PUN':
                    observation_prob = 1.0
                else:
                    observation_prob = 0.0
            
            # this is the case where we encounter a word we havent seen before:
            else:
                #observation_prob = 0.0 #change this?
                smoothing_k = 1e-3
                observation_prob = smoothing_k / (tag_counts[i] + smoothing_k * (len(observation_probabilities[i]) + 1)) # Laplace smoothing

            temp_probs = prob[t - 1] * transition_probabilities[:, i] * observation_prob
            x = np.argmax(temp_probs)
            prob[t, i] = temp_probs[x]
            prev[t, i] = x

    # Get index for tag with highest prob for final word
    best_i_last_word = np.argmax(prob[-1])
    # Store this index into the best_path list
    best_path = [best_i_last_word]

    for t in range(num_observations - 1, 0, -1):
        best_path.append(prev[t, best_path[-1]])

    # to get path from start->end
    best_path.reverse()
    best_path = [TAGS[i] for i in best_path] # convert indices to their respective tags

    # return list of (word,tag) tuples
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
    initial_pr, transition_pr, observation_pr, tag_counts = generate_tables_V2(training_sentences)

    print("Here are the initial probabilities:")
    print(initial_pr)
    print(" ")

    print("here are the transitional probabilities, 3 random rows of it:")
    print(transition_pr[0])
    print(transition_pr[4])
    print(transition_pr[20])
    print(" ")

    # print("here are the observational probabilities:")
    # print(observation_pr)
    # print(" ")


    tagged_sentences = []

    # run veterbi for each sentence and add tagged sentence to list
    for sentence in testing_sentences:
        tagged_sentence = viterbi_algorithm_V2(sentence, initial_pr, transition_pr, observation_pr, tag_counts)
        tagged_sentences.append(tagged_sentence)

    #print(tagged_sentences)
    print("tagging sentences and writing up the output file now.")
    with open(args.outputfile, 'w') as output_file:
        for tagged_sentence in tagged_sentences:
            for word, tag in tagged_sentence:
                output_file.write(f"{word} : {tag}\n")
