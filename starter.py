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

def calculate_posterior(emails):
    """This function should return the probability if an email contains the
    word 'lottery.'
    """
    pass
    

emails["words"] = emails["text"].apply(process_email)