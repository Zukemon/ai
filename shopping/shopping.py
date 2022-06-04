import csv
import sys

# from sklearn.svm import SVC
# from sklearn.linear_model import Perceptron
# from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

# model = Perceptron()


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
       0 - Administrative, an integer
       1 - Administrative_Duration, a floating point number
       2 - Informational, an integer
       3 - Informational_Duration, a floating point number
       4 - ProductRelated, an integer
       5 - ProductRelated_Duration, a floating point number
       6 - BounceRates, a floating point number
       7 - ExitRates, a floating point number
       8 - PageValues, a floating point number
       9 - SpecialDay, a floating point number
       10 - Month, an index from 0 (January) to 11 (December)
       11 - OperatingSystems, an integer
       12 - Browser, an integer
       13 - Region, an integer
       14 - TrafficType, an integer
       15 - VisitorType, an integer 0 (not returning) or 1 (returning)
       16 - Weekend, an integer 0 (if false) or 1 (if true)
       17 - Revenue - "evidence"

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    
    # function for enumerating returning visitors output
    def Visitors():    
        if col[15] == "Returning_Visitor":
            return '1'
        else:
            return '0'

    # function for enumerating returning visitors output
    def Weekend():    
        if col[16] == "TRUE":
            return '1'
        else:
            return '0'

    # enumerating months for evidence output
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_num = enumerate(months)
    month = {k: v for v, k in month_num} 

    with open("shopping.csv") as f:
        reader = csv.reader(f)
        next(reader)
        
        
        # splitting data = [], to evidence and label
        evidence = []
        label = []
        
        # iterating through csv rows by their columns indexes + slices      
        for col in reader:

            int1 = col[0]
            
            flt1 = col[1]
            
            int2 = col[2]
        
            flt2 = col[3]
        
            int3 = col[4]
        
            flt3 = col[5:10]
            
            monthz = (int(month[col[10]]))

            int4 = col[11:15] #+ [int(int5)] + [int(int6)] + [int(int7)] 
            
            
            if col[15]:
                visitors = Visitors()
            

            if col[16]:
                wknd_visitors = Weekend()
                

            # appending clean evidence and label data
            evidence.append([int(int1)] + [float(flt1)] + [int(int2)] + [float(flt2)] + \
                            [int(int3)] + [float(cell) for cell in flt3] + \
                            [int(monthz)] + [int(cell) for cell in int4] + [int(visitors)] + [int(wknd_visitors)]),
            
            label.append(1 if col[17] == "TRUE" else 0)

            labels = label
    # returning evidence and labels for training model
    return evidence, labels
    
    
    # raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # training agent with the KNeighborsClassifier and fit model
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    
    return model
    # raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    
    sensitivity = 0
    specificity = 0

    # initializing polarities
    positive = 0
    negative = 0

    # initializing true positives and negative values
    beta_positives = 0
    beta_negatives = 0

    # measuring label size for iteration
    labeled = len(labels)

    for label in range(labeled):

        # if positive
        if labels[label] == 1:
            positive += 1
            if labels[label] == predictions[label]:
                beta_positives += 1

        # and if negative
        if labels[label] == 0:
            negative += 1
            if labels[label] == predictions[label]:
                beta_negatives += 1

    # measuring positive rate
    sensitivity = beta_positives/ positive
    
    # measuring negative rate
    specificity = beta_negatives/ negative

    return sensitivity, specificity
    
    
    # raise NotImplementedError


if __name__ == "__main__":
    main()
