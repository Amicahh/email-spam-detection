import pandas
import numpy
emails = pandas.read_csv("emails.csv")

def process_email(text):
    text = text.lower()
    return list(numpy.unique(text.split()))

def predict_naive_bayes(word):
    pass


def calculate_priors(emails):
    prior_spam = numpy.sum(emails['spam'] == 1) / len(emails)
    prior_ham = 1 - prior_spam
    return prior_spam, prior_ham

def calculate_posterior(emails, given_words):
    words_in_spam = {}
    words_in_ham = {}
    for _, email in emails.iterrows(): #
        if email["spam"] == 1:
            for word in email["words"]:
                words_in_spam[word] = words_in_spam.get(word, 0) + 1
        else:
            for word in email["words"]:
                words_in_ham[word] = words_in_ham.get(word, 0) + 1
    prior_spam, prior_ham = calculate_priors(emails)
    total_spam_word_count = sum(words_in_spam.values())
    total_ham_word_count = sum(words_in_ham.values())
    
    probability_of_word_given_spam = 1 # initially
    for given_word in given_words:
        count_given_word_in_spam = words_in_spam.get(given_word, 0)
        probability_of_word_given_spam *= (count_given_word_in_spam + 1) / (total_spam_word_count + len(words_in_spam))
        # every word will affect the probability

    probability_of_word_given_ham = 1
    for given_word in given_words:
        count_given_word_in_ham = words_in_ham.get(given_word, 0)
        probability_of_word_given_ham *= (count_given_word_in_ham + 1) / (total_ham_word_count + len(words_in_ham))

    probability_of_observing_given_word = (probability_of_word_given_spam * prior_spam) + (probability_of_word_given_ham * prior_ham)

    posterior = (probability_of_word_given_spam * prior_spam) / probability_of_observing_given_word
    return posterior

    

emails["words"] = emails["text"].apply(process_email)