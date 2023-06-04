####################################################
# Created by: Yusuf Demir                          #
# Course: Machine Learning                         #
# Title: C4.5 Decision Tree Algorithm with Pruning #
####################################################
# In order to run this code, you need to install graphviz
# pip install graphviz
# pip install pandas
# pip install numpy

import pandas as pd 
import numpy as np
from graphviz import Digraph


def entropy(y):
    """
    Calculates the entropy of a dataset.

    Args:
        y (pandas.Series): The target variable.

    Returns:
        float: The entropy of the dataset.
    """
    counts = y.value_counts() 
    probs = counts / y.shape[0] 
    entropy = -np.sum(probs * np.log2(probs)) 
    return entropy

def split_data(data, attribute, value):
    """
    Splits the data into two subsets based on the attribute and value.

    Args:
        data (pandas.DataFrame): The dataset to split.
        attribute (str): The attribute to split on.
        value (float): The value to split on.

    Returns:
        left (pandas.DataFrame): The subset of data where the attribute is less than or equal to the value.
        right (pandas.DataFrame): The subset of data where the attribute is greater than the value.
    """
    left = data[data[attribute] <= value] 
    right = data[data[attribute] > value] 
    return left, right

def information_gain(data, attribute, value):
    """
    Calculates the information gain of a dataset.

    Args:
        data (pandas.DataFrame): The dataset to calculate the information gain for.
        attribute (str): The attribute to split on.
        value (float): The value to split on.

    Returns:
        float: The information gain of the dataset.
    """
    left, right = split_data(data, attribute, value) 
    if left.shape[0] == 0 or right.shape[0] == 0: 
        return 0 
    left_entropy = entropy(left["class"]) 
    right_entropy = entropy(right["class"])
    total_entropy = (left.shape[0] / data.shape[0]) * left_entropy + (right.shape[0] / data.shape[0]) * right_entropy
    return entropy(data["class"]) - total_entropy

class Node:
    """
    A node in a decision tree.
    """
    def __init__(self, data, max_depth):
        self.left = None 
        self.right = None 
        self.attribute = None 
        self.value = None 
        self.label = None
        self.gain = None
        self.entropy = entropy(data["class"])
        self.create_node(data, max_depth)
    
    # this code does include early stopping mechanisms to avoid growing the decision tree too deep (which can be seen as a form of simplified pruning):
    def create_node(self, data, max_depth):
        """
        Creates a node in the decision tree.

        Args:
            data (pandas.DataFrame): The dataset to create the node for.
            max_depth (int): The maximum depth of the tree.

        Returns:
            None
        """
        # label the node left is less than or equal to the value
        if max_depth == 0 or data.shape[0] == 0:
            self.label = data["class"].value_counts().idxmax()
            return
        if data["class"].nunique() == 1:
            self.label = data["class"].unique()[0]
            return
        
        self.attribute, self.value = self.get_best_split(data)
        
        left, right = split_data(data, self.attribute, self.value) 
        self.left = Node(left, max_depth - 1) 
        self.right = Node(right, max_depth - 1)

    def get_best_split(self, data):
        """
        Finds the best attribute and value to split on.

        Args:
            data (pandas.DataFrame): The dataset to find the best split for.

        Returns:
            attribute (str): The attribute to split on.
            value (float): The value to split on.
        """
        max_gain = 0 
        attribute = None 
        value = None 
        for col in data.columns[:-1]: 
            for val in data[col].unique(): 
                gain = information_gain(data, col, val) 
                if gain > max_gain: 
                    max_gain = gain 
                    attribute = col 
                    value = val
                    self.gain = max_gain
                    
        return attribute, value

def predict(node, x):
    """
    Predicts the class of a single data point.

    Args:
        node (Node): The root node of the decision tree.
        x (pandas.Series): The data point to predict the class of.

    Returns:
        str: The predicted class of the data point.
    """ 
    if node.label is not None: 
        return node.label 
    if x[node.attribute] <= node.value: 
        return predict(node.left, x) 
    else: return predict(node.right, x)



# print to screen the tree
def print_tree2(node, level=0):
    """
    Prints the decision tree to the screen.
    
    Args:
        node (Node): The root node of the decision tree.
        level (int): The current level of the tree.
        
    Returns:
        None
    """
    if node.label is not None:
        print("  " * level + str(node.label))
        return
    print("  " * level + str(node.attribute) + " <= " + str(node.value))
    print_tree2(node.left, level + 1)
    print("  " * level + str(node.attribute) + " > " + str(node.value))
    print_tree2(node.right, level + 1)

def predictions(length_of_data, data):
    predictions = []
    for i in range(length_of_data):
        row = data.iloc[i, :-1]
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions


def accuracy(predictions, test):
    """
    Calculates the accuracy of the predictions.

    Args:
        predictions (list): The list of predictions.
        test (pandas.DataFrame): The test dataset.

    Returns:
        float: The accuracy of the predictions.
    """
    correct = 0
    for index,i in enumerate(predictions):
        if i == test.iloc[index,-1]:
            correct += 1
    return correct/len(test)


def TPrate(predictions, test):
    """
    Calculates the true positive rate of the predictions.

    Args:
        predictions (list): The list of predictions.
        test (pandas.DataFrame): The test dataset.

    Returns:
        float: The true positive rate of the predictions.

    """
    true_positive = 0
    false_negative = 0
    for index,i in enumerate(predictions):
        if i == test.iloc[index,-1] and i == "good":
            true_positive += 1
        elif i != test.iloc[index,-1] and i == "bad":
            false_negative += 1
    return true_positive/(true_positive + false_negative)


def TNrate(predictions, test):
    """
    Calculates the true negative rate of the predictions.
    
    Args:
        predictions (list): The list of predictions.
        test (pandas.DataFrame): The test dataset.
        
    Returns:
        float: The true negative rate of the predictions.
    """
    true_negative = 0
    false_positive = 0
    for index,i in enumerate(predictions):
        if i == test.iloc[index,-1] and i == "bad":
            true_negative += 1
        elif i != test.iloc[index,-1] and i == "good":
            false_positive += 1
    return true_negative/(true_negative + false_positive)


def TP_count(predictions, test):
    """
    Calculates the number of true positives in the predictions.
    
    Args:
        predictions (list): The list of predictions.
        test (pandas.DataFrame): The test dataset.
        
    Returns:
        int: The number of true positives in the predictions.
    """
    true_positive = 0
    for index,i in enumerate(predictions):
        if i == test.iloc[index,-1] and i == "good":
            true_positive += 1
    return true_positive


def TN_count(predictions, test):
    """
    Calculates the number of true negatives in the predictions.

    Args:
        predictions (list): The list of predictions.
        test (pandas.DataFrame): The test dataset.

    Returns:
        int: The number of true negatives in the predictions.
    """
    true_negative = 0
    for index,i in enumerate(predictions):
        if i == test.iloc[index,-1] and i == "bad":
            true_negative += 1
    return true_negative


# Using graphviz to visualize the tree
def print_tree(node, dot=None):
    """
    Prints the decision tree to a file.

    Args:
        node (Node): The root node of the decision tree.
        dot (graphviz.Digraph): The graph to add the tree to.

    Returns:
        None
    """
    if node.label is not None:
        if str(node.label) == "good":
            dot.node(str(node), str(node.label).capitalize(),shape="ellipse",style="filled",fillcolor="lightgreen")
            return
        else:
            dot.node(str(node), str(node.label).capitalize(),shape="ellipse",style="filled",fillcolor="lightpink")
            return
    dot.node(str(node), str(node.attribute) + " <= " + str(node.value) + "\n" 
             +"Entropy = " +str(round(node.entropy,3)) + "\n" 
             +"Information Gain = " +str(round(node.gain,3)),shape="box",style="filled",fillcolor="lightblue")
    print_tree(node.left, dot=dot)
    dot.edge(str(node), str(node.left),xlabel="True")
    print_tree(node.right, dot=dot)
    dot.edge(str(node), str(node.right),label="False")

# Post pruning
def post_pruning(node, test):
    """
    Prunes the decision tree.

    Args:
        node (Node): The root node of the decision tree.
        test (pandas.DataFrame): The test dataset.

    Returns:
        Node: The root node of the pruned decision tree.
    """
    if node.label is not None:
        return node
    else:
        node.left = post_pruning(node.left, test)
        node.right = post_pruning(node.right, test)
        if node.left.label is not None and node.right.label is not None:
            if node.left.label == node.right.label:
                node.label = node.left.label
                node.left = None
                node.right = None
                return node
            else:
                predictions_test = predictions(len(test),test)
                predictions_train = predictions(len(train),train)
                if accuracy(predictions_test, test) > accuracy(predictions_train, train):
                    node.label = node.left.label
                    node.left = None
                    node.right = None
                    return node
                else:
                    return node
        else:
            return node


print("Starting..")
train = pd.read_csv("trainSet.csv")
test = pd.read_csv("testSet.csv")


# create the tree
tree = Node(train, max_depth=7)

# print the tree to the screen
print_tree2(tree)

# make predictions
predictions_test = predictions(len(test),test)
predictions_train = predictions(len(train),train)

# prune the tree
tree = post_pruning(tree, test)

# print to file the tree
dot = Digraph(comment='Decision Tree')
print_tree(tree, dot=dot)
dot.render('decision_tree', format='png')

# pop up the tree
from PIL import Image
img = Image.open("decision_tree.png")
img.show()




# print results
print("Egitim (Train) sonucu:")
print("Accuracy: ",accuracy(predictions_train, train))
print("TP rate: ",TPrate(predictions_train, train))
print("TN rate: ",TNrate(predictions_train, train))
print("TP count: ",TP_count(predictions_train, train))
print("TN count: ",TN_count(predictions_train, train))

print("\nSinama (Test) sonucu:")
print("Accuracy: ",accuracy(predictions_test, test))
print("TP rate: ",TPrate(predictions_test, test))
print("TN rate: ",TNrate(predictions_test, test))
print("TP count: ",TP_count(predictions_test, test))
print("TN count: ",TN_count(predictions_test, test))

# Write results to txt file
with open("results.txt", "w") as f:
    f.write("Egitim (Train) sonucu:\n")
    f.write("Accuracy: " + str(accuracy(predictions_train, train))+"\n")
    f.write("TP rate: " + str(TPrate(predictions_train, train))+"\n")
    f.write("TN rate: " + str(TNrate(predictions_train, train))+"\n")
    f.write("TP count: " + str(TP_count(predictions_train, train))+"\n")
    f.write("TN count: " + str(TN_count(predictions_train, train))+"\n")
    f.write("\nSinama (Test) sonucu:"+"\n\n")
    f.write("Accuracy: " + str(accuracy(predictions_test, test))+"\n")
    f.write("TP rate: " + str(TPrate(predictions_test, test))+"\n")
    f.write("TN rate: " + str(TNrate(predictions_test, test))+"\n")
    f.write("TP count: " + str(TP_count(predictions_test, test))+"\n")
    f.write("TN count: " + str(TN_count(predictions_test, test))+"\n")
