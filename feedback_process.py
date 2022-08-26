import pandas as pd
import csv
import nltk

#import CSV file as a Pandas dataframe
data = pd.read_csv('Test_feedback.csv', index_col = 0)

#Separate English and French data
data_en = data[data['Language'].str.contains("en", na=False)]
data_fr = data[data['Language'].str.contains("fr", na=False)]

#Word lists
word_list_en = data_en["Task Why Not Comment"].tolist()
word_list_en = [str(i) for i in word_list_en]
all_words_en = ' '.join([str(elem) for elem in word_list_en])


word_list_fr = data_fr["Task Why Not Comment"].tolist()
word_list_fr = [str(i) for i in word_list_fr]
all_words_fr = ' '.join([str(elem) for elem in word_list_fr])


#tokenize words
tokenizer = nltk.RegexpTokenizer(r"\w+")
tokens_en = tokenizer.tokenize(all_words_en)
words_en = []
for word in tokens_en:
        words_en.append(word.lower())

tokens_fr = tokenizer.tokenize(all_words_fr)
words_fr = []
for word in tokens_fr:
        words_fr.append(word.lower())

#remove nan
words_en = list(filter(('nan').__ne__, words_en))
words_fr = list(filter(('nan').__ne__, words_fr))


#remove English stop words to get most frequent words
nltk.download('stopwords')
sw = nltk.corpus.stopwords.words('english')
sw.append('covid')
sw.append('19')
words_ns_en = []
for word in words_en:
        if word not in sw:
            words_ns_en.append(word)


from nltk import FreqDist
fdist1 = FreqDist(words_ns_en)
most_common_en = fdist1.most_common(10)
