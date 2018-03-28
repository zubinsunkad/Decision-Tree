#!/usr/bin/env python

import argparse
import csv
import math
import copy
import random


'''
Below is an input parser function to take commandline parameters.
'''

def input_parser():
    parser = argparse.ArgumentParser(description='Program for decision tree classification.', epilog = 'Sample commandline - $python DecisionTree.py --k 1 --l 2 --training "training.csv" --to_print "yes"')
    parser.add_argument('--k', type=int, help='please provide k value.', default = 1,action='store')
    parser.add_argument('--l', type=int, help='please provide l value.', default = 1,action='store')
    parser.add_argument('--training', type=str, help='please provide training data path.',default="training_set.csv",action='store')
    parser.add_argument('--validation', type=str, help='please provide validation data path.', default="validation_set.csv",action='store')
    parser.add_argument('--test', type=str, help='please provide testing data path.', default = "test_set.csv",action='store')
    parser.add_argument('--to_print', type=str, help='please provide yes or no to print.', default = "no",action='store')

    args = parser.parse_args()
    return args

"""
Function below is to Count number of values. Used in built trees and entropy for checking uniqueness.
"""
def uniquecounts(rows):
    results = {}
    for row in rows:
       
        r = row[len(row) - 1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results

"""
Below is the function to calculate Entropy
"""
def entropy(rows):
   
    results = uniquecounts(rows)
    entropy = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        entropy = entropy - p * math.log(p,2)
    return entropy



'''
Below is a function to calculate variance impurity.

'''
def varianceImpurity(rows):
    if len(rows) == 0: return 0
    results = uniquecounts(rows)
    total_samples = len(rows)
    variance_impurity = (results['0'] * results['1']) / (total_samples ** 2)
    return variance_impurity

'''
Function to split values.
'''
def div(rows, column, value):
    split = None
    if isinstance(value, int):
        split = lambda row: row[column] >= value
    else:
        split = lambda row: row[column] == value

    set1 = [row for row in rows if split(row)]
    set2 = [row for row in rows if not split(row)]
    return (set1, set2)

'''
Class Node is used to create the tree.  
'''
class Node:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb

'''
Function to build tree amd split based on varience impurity or entropy calculated.
'''
def buildtree(rows, scoref=entropy):
    if len(rows) == 0: return Node()
    current_score = scoref(rows)
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    column_count = len(rows[0]) - 1
    for col in range(0, column_count): 
        global column_values
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        for value in column_values.keys():
            (set1, set2) = div(rows, col, value)

            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return Node(col=best_criteria[0], value=best_criteria[1],
                            tb=trueBranch, fb=falseBranch)
    else:
        return Node(results=uniquecounts(rows))

'''
Function to print tree as per required format in assignment.
'''

def print_tree(tree, header_data, indent):
    if tree.results != None:
        for key in tree.results:
            print(str(key))
    else:
        print("")
        print(indent + str(header_data[tree.col]) + ' = ' + str(tree.value) + ' : ', end="")
        print_tree(tree.tb, header_data, indent + '  |')

        print(indent + str(header_data[tree.col]) + ' = ' + str(int(tree.value) ^ 1) + ' : ', end="")
        print_tree(tree.fb, header_data, indent + '  |')

'''
Function to calculate accuracy.
'''
def accuracy(rows, tree):
    correct_predictions = 0
    for row in rows:
        pred_val = classify(row, tree)
        if row[-1] == pred_val:
            correct_predictions += 1
    accuracy = 100 * correct_predictions / len(rows)
    return accuracy
'''
Function to classify as the name suggests.
'''
def classify(observation, tree):
    if tree.results != None:
        for key in tree.results:
            predicted_value = key
        return predicted_value
    else:
        v = observation[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        predicted_value = classify(observation, branch)
    return predicted_value

def list_nodes(nodes, tree, count):
    if tree.results != None:
        return nodes, count
    count += 1
    nodes[count] = tree
    (nodes, count) = list_nodes(nodes, tree.tb, count)
    (nodes, count) = list_nodes(nodes, tree.fb, count)
    return nodes, count
    
def count_class_occurence(tree, class_occurence):
    if tree.results != None:
        for key in tree.results:
            class_occurence[key] += tree.results[key]
        return class_occurence
    left_branch_occurence = count_class_occurence(tree.fb, class_occurence)
    right_branch_occurence = count_class_occurence(tree.tb, left_branch_occurence)
    return right_branch_occurence
'''
Replace Tree using Pruning Algorithm.
'''
def findAndReplaceSubtree(tree_copy, subtree_to_replace, subtree_to_replace_with):
    if (tree_copy.results != None):
        return tree_copy
    if (tree_copy == subtree_to_replace):
        tree_copy = subtree_to_replace_with
        return tree_copy
    tree_copy.fb = findAndReplaceSubtree(tree_copy.fb, subtree_to_replace, subtree_to_replace_with)
    tree_copy.tb = findAndReplaceSubtree(tree_copy.tb, subtree_to_replace, subtree_to_replace_with)
    return tree_copy

'''
Function to prune tree
'''
def prune_tree(tree, l, k, data):
    tree_best = tree
    best_accuracy = accuracy(data, tree)
    tree_copy = None
    for i in range(1, l):
        n = random.randint(1, k)
        tree_copy = copy.deepcopy(tree)
        for j in range(1, n):
            (nodes, initial_count) = list_nodes({}, tree_copy, 0)
            if (initial_count > 0):
                p = random.randint(1, initial_count)   
                subtree_p = nodes[p]
                class_occurence = {'0': 0, '1': 0}
                count = count_class_occurence(subtree_p, class_occurence)
                if count['0'] > count['1']:
                    count['0'] = count['0'] + count['1']
                    count.pop('1')
                    subtree_p = Node(results=count)
                else:
                    count['1'] = count['0'] + count['1']
                    count.pop('0')
                    subtree_p = Node(results=count)
                tree_copy = findAndReplaceSubtree(tree_copy, nodes[p], subtree_p)
        curr_accuracy = accuracy(data, tree_copy)
        if (curr_accuracy > best_accuracy):
            best_accuracy = curr_accuracy
            tree_best = tree_copy
        else:
            pass
    
    return tree_best, best_accuracy

'''
Below is the main function.
'''

def main():
        
        args = input_parser()
        l = args.l
        k = args.k
        train = args.training
        val_file = args.validation
        test_filename = args.test
        to_print = args.to_print
        try:
            with open(train, newline='', encoding='utf_8') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                header_data = next(spamreader)
                train_training_data = list(spamreader)
            with open(val_file, newline='', encoding='utf_8') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                validation_training_data = list(spamreader)
        except Exception as e:
            print(e)
        '''
        Above try and except is if program tries to open a file when it is not there and can be used for debugging especially if logging is used.
        ''' 
        with open(test_filename, newline='', encoding='utf_8') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            test_training_data = list(spamreader)

            l_combination = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            k_combination = [9, 7, 1, 3, 4, 6, 5, 9, 8, 2]
            learned_tree_IG = buildtree(train_training_data, scoref=entropy)
            
            print("Information gain")
            if(to_print.lower() == "yes"):
                print_tree(learned_tree_IG, header_data, '')
            else: 
                pass
            train_accuracy = accuracy(train_training_data, learned_tree_IG)
            print("Accuracy for Training set : ", train_accuracy)
            validation_accuracy = accuracy(validation_training_data, learned_tree_IG)
            print("Accuracy for Validation dataset : ", validation_accuracy)
            test_accuracy = accuracy(test_training_data, learned_tree_IG)
            print("Accuracy for Test data : ", test_accuracy)
            (pruned_best_tree_validation, pruned_best_accuracy_validation) = prune_tree(learned_tree_IG, l, k,validation_training_data)
            print("Validation data accuracy with tree - pruning : ", pruned_best_accuracy_validation)
            (pruned_best_tree_test, pruned_best_accuracy_test) = prune_tree(learned_tree_IG, l, k, test_training_data)
            if (to_print.lower() == "yes"):
                print_tree(pruned_best_tree_test, header_data, '')
            print("Test data accuracy with pruning : ", pruned_best_accuracy_test)
            print("Accuracies for 10 combinations of l and k :")
            for l_val, k_val in  zip(l_combination, k_combination):
                (pruned_best_tree_test, pruned_best_accuracy_test) = prune_tree(learned_tree_IG, l_val, k_val,test_training_data)
                print("with pruning l = ", l_val," and k = " , k_val," : ", pruned_best_accuracy_test)

#Variable names with VI suffix refer to variance impurity heuristic.

            learned_tree_var = buildtree(train_training_data, scoref=varianceImpurity)
            print("Variance Impurity")
            if (to_print.lower() == "yes"):
                print_tree(learned_tree_var, header_data, '')
            train_accuracy_VI = accuracy(train_training_data, learned_tree_var)
            print("Accuracy for Training dataset : ", train_accuracy_VI)
            validation_accuracy_VI = accuracy(validation_training_data, learned_tree_var)
            print("Accuracy for Validation set: ", validation_accuracy_VI)
            test_accuracy_VI = accuracy(test_training_data, learned_tree_var)
            print("Test data accuracy : ", test_accuracy_VI)
            (pruned_best_tree_validation_VI, pruned_best_accuracy_validation_VI) = prune_tree(learned_tree_var, l, k, validation_training_data)
            print("Validation data accuracy with pruning: ", pruned_best_accuracy_validation_VI)
            (pruned_best_tree_test_VI, pruned_best_accuracy_test_VI) = prune_tree(learned_tree_var, l, k, test_training_data)
            if (to_print.lower() == "yes"):
                print_tree(pruned_best_tree_test_VI, header_data, '')
            print("Test data accuracy with pruning : ", pruned_best_accuracy_test_VI)
           
           #calculating accuracy for test data with l & k combination. 
            print("Calculating acuracies of test data, 10 combinations of l and k :")
            for l_val, k_val in zip(l_combination, k_combination):
                (pruned_best_tree_test_VI, pruned_best_accuracy_test_VI) = prune_tree(learned_tree_var, l_val, k_val,test_training_data)
                print("l = ", l_val, " and k = ", k_val," : ", pruned_best_accuracy_test_VI)
                for l_val, k_val in zip(l_combination, k_combination):
                    (pruned_best_tree_test, pruned_best_accuracy_test) = prune_tree(learned_tree_IG, l_val, k_val,test_training_data)
                    test_accuracy_str_i = "l = ", l_val, " and k = ", k_val, " : ", pruned_best_accuracy_test             
                for l_val, k_val in zip(l_combination, k_combination):
                    (pruned_best_tree_test_VI, pruned_best_accuracy_test_VI) = prune_tree(learned_tree_var, l_val, k_val,test_training_data)
                    test_accuracy_str_VI_i = "l = ", l_val, " and k = ", k_val," : ", pruned_best_accuracy_test_VI


if __name__ == "__main__":
    main()