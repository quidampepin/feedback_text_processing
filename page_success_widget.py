#import libraries
import pandas as pd
import csv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud

#import CSV file as a Pandas dataframe
data = pd.read_csv('page_success_june_17.csv', index_col = 0)

    #Separate English and French data
data_en = data[data['Page URL'].str.contains("/en", na=False)]

data_fr = data[data['Page URL'].str.contains("/fr", na=False)]

    #look at what's wrong
what = data["What's wrong"]

what_en = data_en["What's wrong"]
what_fr = data_fr["What's wrong"]


#plot what's wrong

plt.rcParams['figure.figsize'] = (10, 6)
plt.gcf().subplots_adjust(left=0.25)
what.value_counts().sort_values().plot.barh(title = 'Feedback by reason', x='Reason', y='Number of occurrences')
plt.savefig('feedback_by_reason.png')
plt.clf()

data = data[data.Topic != 'No details']#plot by task
tasks = data["Topic"].str.split(", ", n = 3, expand = True)
tasks = tasks.apply(pd.Series.value_counts)
tasks = tasks.fillna(0)
tasks = tasks[0] + tasks[1]
tasks = tasks.astype(int)
tasks = tasks.sort_values(ascending = False)
tasks = tasks[0:30]
plt.rcParams['figure.figsize'] = (14, 8)
plt.gcf().subplots_adjust(left=0.30)
tasks.sort_values().plot.barh(title = 'Top 30 tasks', x='Reason', y='Number of occurrences')
plt.savefig('feedback_by_task.png')
plt.clf()

#analyzing  words
word_list_en = data_en["Details"].tolist()
word_list_en = [str(i) for i in word_list_en]
all_words_en = ' '.join([str(elem) for elem in word_list_en])

word_list_fr = data_fr["Details"].tolist()
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

#Plot English most common words
from nltk import FreqDist
fdist1 = FreqDist(words_ns_en)
most_common_en = fdist1.most_common(50)
most_common_df = pd.DataFrame(most_common_en, columns = ['Word', 'Count'])
most_common_df.plot.barh(title = 'Most frequent words - English - All feedback', x='Word',y='Count')
plt.rcParams['figure.figsize'] = (14, 8)
plt.gcf().subplots_adjust(left=0.20)
plt.savefig('frequent_words_en_all.png')
plt.clf()


#WordCloud English
word_cloud_en = ' '.join(words_ns_en)

wordcloud = WordCloud(max_font_size=40).generate(word_cloud_en)

import matplotlib.pyplot as plt

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
plt.savefig('word_cloud_en.png')
plt.clf()

#remove French stop words

swf = nltk.corpus.stopwords.words('french')
swf.append('covid')
swf.append('19')
swf.append('a')
swf.append('si')
swf.append('avoir')
swf.append('savoir')
swf.append('combien')
swf.append('être')
swf.append('où')
swf.append('comment')
swf.append('puis')
swf.append('peuvent')
swf.append('fait')
swf.append('aucun')
swf.append('bonjour')
swf.append('depuis')
swf.append('chez')
swf.append('faire')
swf.append('peut')
swf.append('plus')
swf.append('veux')
swf.append('dois')
swf.append('doit')
swf.append('dit')
swf.append('merci')
swf.append('cela')
swf.append('pouvons')
swf.append('pouvaient')
swf.append('vers')

words_ns_fr = []
for word in words_fr:
        if word not in swf:
            words_ns_fr.append(word)

#plot most frequent French words
fdist1 = FreqDist(words_ns_fr)
most_common_fr = fdist1.most_common(50)
most_common_df_fr = pd.DataFrame(most_common_fr, columns = ['Mot', 'Nombre'])
most_common_df_fr.plot.barh(title = 'Mots les plus fréquents - Toute la rétroaction - Français', x='Mot',y='Nombre')
plt.rcParams['figure.figsize'] = (14, 8)
plt.gcf().subplots_adjust(left=0.20)
plt.savefig('frequent_words_fr_all.png')
plt.clf()


#WordCloud French

word_cloud_fr = ' '.join(words_ns_fr)

wordcloud = WordCloud(max_font_size=40).generate(word_cloud_fr)

import matplotlib.pyplot as plt

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
plt.savefig('word_cloud_fr.png')
plt.clf()


#English bigrams

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
bcf = BigramCollocationFinder.from_words(words_en)
from nltk.corpus import stopwords
stopset = sw
filter_stops = lambda w: len(w) < 3 or w in stopset
bcf.apply_word_filter(filter_stops)
bcf_list = bcf.nbest(BigramAssocMeasures.likelihood_ratio, 20)
bcf_joint_list = []
for words in bcf_list:
        bcf_joint_list.append(' '.join(words))

#save list in txt file
with open('bigrams_en.txt', 'w') as filehandle:
        for bigrams in bcf_joint_list:
            filehandle.write('%s\n' % bigrams)

#English trigrams
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
tcf = TrigramCollocationFinder.from_words(words_en)
tcf.apply_word_filter(filter_stops)
tcf_list = tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 20)
tcf_joint_list = []
for words in tcf_list:
        tcf_joint_list.append(' '.join(words))


with open('trigrams_en.txt', 'w') as filehandle:
        for trigrams in tcf_joint_list:
            filehandle.write('%s\n' % trigrams)


#French bigrams
bcffr = BigramCollocationFinder.from_words(words_fr)
from nltk.corpus import stopwords
stopsetfr = swf
filter_stopsfr = lambda w: len(w) < 3 or w in stopsetfr
bcffr.apply_word_filter(filter_stopsfr)
bcffr_list = bcffr.nbest(BigramAssocMeasures.likelihood_ratio, 20)
bcffr_joint_list = []
for words in bcffr_list:
        bcffr_joint_list.append(' '.join(words))


with open('bigrams_fr.txt', 'w') as filehandle:
        for bigrams in bcffr_joint_list:
            filehandle.write('%s\n' % bigrams)


#French trigrams

from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
tcffr = TrigramCollocationFinder.from_words(words_fr)
tcffr.apply_word_filter(filter_stopsfr)
tcffr_list = tcffr.nbest(TrigramAssocMeasures.likelihood_ratio, 20)
tcffr_joint_list = []
for words in tcffr_list:
        tcffr_joint_list.append(' '.join(words))


with open('trigrams_fr.txt', 'w') as filehandle:
        for trigrams in tcffr_joint_list:
            filehandle.write('%s\n' % trigrams)


#look at data by what's wrong value_counts
#most common words info is missing
data_missing_en = data_en[data_en["What's wrong"].str.contains("missing", na=False)]

word_list_missing_en = data_missing_en["Details"].tolist()
word_list_missing_en = [str(i) for i in word_list_missing_en]
all_words_missing_en = ' '.join([str(elem) for elem in word_list_missing_en])

tokenizer = nltk.RegexpTokenizer(r"\w+")
tokens_missing_en = tokenizer.tokenize(all_words_missing_en)
words_missing_en = []
for word in tokens_missing_en:
        words_missing_en.append(word.lower())

words_missing_en = list(filter(('nan').__ne__, words_missing_en))

words_missing_ns_en = []
for word in words_missing_en:
        if word not in sw:
            words_missing_ns_en.append(word)

from nltk import FreqDist
fdist1 = FreqDist(words_missing_ns_en)
most_common_missing_en = fdist1.most_common(50)
most_common_missing_df = pd.DataFrame(most_common_missing_en, columns = ['Word', 'Count'])
most_common_missing_df.plot.barh(title = 'Most frequent words - English - Information is missing', x='Word',y='Count')
plt.rcParams['figure.figsize'] = (14, 8)
plt.gcf().subplots_adjust(left=0.20)
plt.savefig('frequent_missing_en.png')
plt.clf()

# most common words clear
data_clear_en = data_en[data_en["What's wrong"].str.contains("clear", na=False)]

word_list_clear_en = data_clear_en["Details"].tolist()
word_list_clear_en = [str(i) for i in word_list_clear_en]
all_words_clear_en = ' '.join([str(elem) for elem in word_list_clear_en])

tokenizer = nltk.RegexpTokenizer(r"\w+")
tokens_clear_en = tokenizer.tokenize(all_words_clear_en)
words_clear_en = []
for word in tokens_clear_en:
        words_clear_en.append(word.lower())

words_clear_en = list(filter(('nan').__ne__, words_clear_en))

words_clear_ns_en = []
for word in words_clear_en:
        if word not in sw:
            words_clear_ns_en.append(word)

from nltk import FreqDist
fdist1 = FreqDist(words_clear_ns_en)
most_common_clear_en = fdist1.most_common(50)
most_common_clear_df = pd.DataFrame(most_common_clear_en, columns = ['Word', 'Count'])
most_common_clear_df.plot.barh(title = 'Most frequent words - English - Information is not clear', x='Word',y='Count')
plt.rcParams['figure.figsize'] = (14, 8)
plt.gcf().subplots_adjust(left=0.20)
plt.savefig('frequent_not_clear_en.png')
plt.clf()




#separate feedback by topic
data_en_topic= data_en.drop(columns=['Date/time received', 'Page Title', 'Page URL', 'Y/N', "What's wrong", 'Personal info (Y/N)', 'Notes', 'Blank', 'Test - auto-topic generator'])

data_en_topic = data_en_topic[data_en_topic.Topic != 'No details']
data_en_topic = data_en_topic[data_en_topic.Topic != 'Feedback unclear']
data_en_topic = data_en_topic.dropna()


data_fr_topic = data_fr.drop(columns=['Date/time received', 'Page Title', 'Page URL', 'Y/N', "What's wrong", 'Personal info (Y/N)', 'Notes', 'Blank', 'Test - auto-topic generator'])

data_fr_topic = data_fr_topic[data_fr_topic.Topic != 'No details']
data_fr_topic = data_fr_topic[data_fr_topic.Topic != 'Feedback unclear']
data_fr_topic = data_fr_topic.dropna()

#separate topic in several columns

data_en_topic = pd.concat([data_en_topic[['Details']], data_en_topic['Topic'].str.split(', ', expand=True)], axis=1)
data_fr_topic = pd.concat([data_fr_topic[['Details']], data_fr_topic['Topic'].str.split(', ', expand=True)], axis=1)


#create dictionary
dict_en = {}
for topic, topic_df_en in data_en_topic.groupby(0):
    dict_en[topic] = ' '.join(topic_df_en['Details'].tolist())


for topic, topic_df_en in data_en_topic.groupby(1):
    dict_en[topic] = ' '.join(topic_df_en['Details'].tolist())


dict_fr = {}
for topic, topic_df_fr in data_fr_topic.groupby(0):
    dict_fr[topic] = ' '.join(topic_df_fr['Details'].tolist())


for topic, topic_df_fr in data_fr_topic.groupby(1):
    dict_fr[topic] = ' '.join(topic_df_fr['Details'].tolist())


#tokenize by topic
for value in dict_en:
    dict_en[value] = tokenizer.tokenize(dict_en[value])

for value in dict_fr:
    dict_fr[value] = tokenizer.tokenize(dict_fr[value])

#analyze words in comparison wth the others - tf-idf analysis
#separate topics and words in 2 lists
topic_list_en = []
for keys in dict_en.keys():
    topic_list_en.append(keys)

topic_list_fr = []
for keys in dict_fr.keys():
    topic_list_fr.append(keys)

topic_words_en = []
for values in dict_en.values():
    topic_words_en.append(values)

topic_words_fr = []
for values in dict_fr.values():
    topic_words_fr.append(values)


#lower case, remove stop words and lemmatize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
topic_words_en = [[word.lower() for word in value] for value in topic_words_en]
topic_words_en = [[lemmatizer.lemmatize(word) for word in value] for value in topic_words_en]
topic_words_en = [[word for word in value if word not in sw] for value in topic_words_en]

topic_words_fr = [[word.lower() for word in value] for value in topic_words_fr]
topic_words_fr = [[lemmatizer.lemmatize(word) for word in value] for value in topic_words_fr]
topic_words_fr = [[word for word in value if word not in swf] for value in topic_words_fr]

#Create dictionary of words in GenSim
from gensim.corpora.dictionary import Dictionary
dictionary_en = Dictionary(topic_words_en)
dictionary_fr = Dictionary(topic_words_fr)

#create corpus
corpus_en = [dictionary_en.doc2bow(topic) for topic in topic_words_en]
corpus_fr = [dictionary_fr.doc2bow(topic) for topic in topic_words_fr]

#code to for tf-idf for one topics
from gensim.models.tfidfmodel import TfidfModel
tfidf_en = TfidfModel(corpus_en)

tfidf_fr = TfidfModel(corpus_fr)

tfidf_weights_en = [sorted(tfidf_en[doc], key=lambda w: w[1], reverse=True) for doc in corpus_en]
tfidf_weights_fr = [sorted(tfidf_fr[doc], key=lambda w: w[1], reverse=True) for doc in corpus_fr]

weighted_words_en = [[(dictionary_en.get(id), weight) for id, weight in ar] for ar in tfidf_weights_en]
weighted_words_fr = [[(dictionary_fr.get(id), weight) for id, weight in ar] for ar in tfidf_weights_fr]

import csv
with open('weighted_words_en.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(topic_list_en, weighted_words_en))

with open('weighted_words_fr.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(topic_list_fr, weighted_words_fr))


#Autotagging
#build training and testing sets from manually tagged data_en
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

data_en_topic_1 = data_en_topic.drop(columns=[1])
data_en_topic_1  = data_en_topic_1.rename(columns={'Details': 'Feedback', 0: 'label'})
y = data_en_topic_1.label

X_train, X_test, y_train, y_test = train_test_split(data_en_topic_1["Feedback"], y, test_size=0.23, random_state=53)
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB(0.1)
nb_classifier.fit(tfidf_train, y_train)
pred = nb_classifier.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print(score)

pred_topic_en = data_en[data_en.Topic != 'No details']
X_pred_en = pred_topic_en['Details']
tfidf_pred_en = tfidf_vectorizer.transform(X_pred_en)
pred_en = nb_classifier.predict(tfidf_pred_en)
pred_topic_en["Predicted_topic"] = pred_en
pred_topic_en.to_csv('predicted_en.csv')


data_fr_topic_1 = data_fr_topic.drop(columns=[1])
data_fr_topic_1  = data_fr_topic_1.rename(columns={'Details': 'Feedback', 0: 'label'})
y = data_fr_topic_1.label

X_train, X_test, y_train, y_test = train_test_split(data_fr_topic_1["Feedback"], y, test_size=0.1, random_state=53)
tfidf_vectorizer = TfidfVectorizer(stop_words=swf, max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB(0.6)
nb_classifier.fit(tfidf_train, y_train)
pred = nb_classifier.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print(score)

pred_topic_fr = data_fr[data_fr.Topic != 'No details']
X_pred_fr = pred_topic_fr['Details']
tfidf_pred_fr = tfidf_vectorizer.transform(X_pred_fr)
pred_fr = nb_classifier.predict(tfidf_pred_fr)
pred_topic_fr["Predicted_topic"] = pred_fr
pred_topic_fr.to_csv('predicted_fr.csv')
