import nltk
import sys
# from nltk.tokenize import RegexpTokenizer

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S ConjP S | VP NP

AdjP -> Adj | Adj AdjP | Det AdjP
AdvP -> Adv | Adv AdvP 
ConjP -> Conj 
NP -> N | Det N | Adj NP | N PP | AdjP N | Det N Adv
PP -> PP NP 
VP -> V | VP P | VP NP | Adv V | V Adv
"""


grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # tokenize sentence to words for grammar rule chunks and tree graph
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    token = tokenizer.tokenize(str(sentence).lower())

    return token

    # raise NotImplementedError


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    # adding phrase chunks as per tokenization following grammar rules
    np = []
    for subtree in tree.subtrees(check):       
        np.append(subtree)  
    return np

    # raise NotImplementedError


def check(subtree):
    """
    Checks for 'NP' labels without subtrees...
    """
    # checking for proper subtree, according to specification for np_chunks
    if subtree.label() == "NP":
        return True
    
    return False



if __name__ == "__main__":
    main()
