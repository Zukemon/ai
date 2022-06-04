import nltk
import sys
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # optimizing directory path for loading files
    data = dict()
    reader = os.listdir(directory)
     
    for row in reader:
        with open(os.path.join(directory, row)) as f:            
                # name = row["name"]
            data[row] = f.read()
    return data
    
    
    # raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    # # tokenize sentence to words for grammar rule chunks and tree graph
    # tokenizer = nltk.RegexpTokenizer(r'\w+')
    # token = tokenizer.tokenize(str(document).lower())
    #
    # return token 

    filt_words = []

    stopWords = set(nltk.corpus.stopwords.words('english'))

    token = nltk.tokenize.word_tokenize(document.lower()) 

    for w in token:
        if w not in stopWords:
            filt_words.append(w)

    return filt_words

    # raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # initializing dictionary for returning idfs
    idfs = dict()

    # creating a set for iterating word iteration
    words = set()

    for filename in documents:
        filename = documents[filename]
        words.update(set(filename))

    for word in words:
        count = 0
        
        for document in documents.values():
            if word in document:
                count += 1
        
        idfs[word] = math.log(len(documents) / count)

    return idfs

    # raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # finding top ranked files
    tf_ranks = []
    for filename in files:
        count = 0
        for item in query:
            # rank increament + multipling item count with item idf value
            count += files[filename].count(item) * idfs[item]
        # appending and sorting file names by attribute of rank
        tf_ranks.append((filename, count))
    tf_ranks.sort(key=lambda tfdi: tfdi[1], reverse=True)
    # returning top files at nth rank
    top_i = [x[0] for x in tf_ranks[:n]]
    return top_i
    
    # raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # finding top ranked sentences...
    tf_ranks = []
    for sentence in sentences:
        count = 0
        for item in query:
            # rank increament + multipling item count with item idf value
            count += sentences[sentence].count(item) * idfs[item]
        # appending and sorting sentences by rank attribute
        tf_ranks.append((sentence, count))
    tf_ranks.sort(key=lambda tfdi: tfdi[1], reverse=True)
    # returning top sentences at nth rank.
    top_s = [x[0] for x in tf_ranks[:n]]
    return top_s
    
    # raise NotImplementedError


if __name__ == "__main__":
    main()
