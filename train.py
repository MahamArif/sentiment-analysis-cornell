import tarfile
import os
import nltk
from nltk.corpus import stopwords
import string, re
from collections import Counter
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import precision_score, roc_auc_score, recall_score, f1_score, balanced_accuracy_score

file = tarfile.open('review_polarity.tar.gz')
file.extractall('./reviews')
file.close()

review_dataset_path="./reviews/txt_sentoken"
print(os.listdir(review_dataset_path))

pos_review_folder_path=review_dataset_path+"/"+"pos"
neg_review_folder_path=review_dataset_path+"/"+"neg"

pos_review_file_names=os.listdir(pos_review_folder_path)
neg_review_file_names=os.listdir(neg_review_folder_path)

def load_review_from_textfile(path):
    file=open(path,"r")
    review=file.read()
    file.close()
    return review

def get_data_target(folder_path, file_names, review_type):
    data=list()
    target =list()
    for file_name in file_names:
        full_path = folder_path + "/" + file_name
        review =load_review_from_textfile(path=full_path)
        data.append(review)
        target.append(review_type)
    return data, target

pos_data, pos_target = get_data_target(folder_path=pos_review_folder_path,
               file_names=pos_review_file_names,
               review_type="positive")
print("Positive data ve target builded...")
print("positive data length:",len(pos_data))
print("positive target length:",len(pos_target))

neg_data, neg_target = get_data_target(folder_path = neg_review_folder_path,
                                      file_names= neg_review_file_names,
                                      review_type="negative")
print("Negative data ve target builded..")
print("negative data length :",len(neg_data))
print("negative target length :",len(neg_target))

data = pos_data + neg_data
target_ = pos_target + neg_target
print("Positive and Negative sets concatenated")
print("data length :",len(data))
print("target length :",len(target_))

le = LabelEncoder()
le.fit(target_)
target = le.transform(target_)
print("Target labels transformed to number...")

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, stratify=target, random_state=24)
print("Dataset splited into train and test parts...")
print("train data length  :",len(X_train))
print("train target length:",len(y_train))
print()
print("test data length  :",len(X_test))
print("test target length:",len(y_test))

nltk.download('stopwords')

text_len = np.vectorize(len)
text_lengths = text_len(X_train)

mean_review_length =int(text_lengths.mean())
print("Mean length of reviews   :",mean_review_length)    
print("Minimum length of reviews:",text_lengths.min())
print("Maximum length of reviews:",text_lengths.max())

replace_by = [("."," "), ("?"," "), (","," "), ("!"," "),(":"," "),(";"," ")]
FACTOR = 8
limited_text_length = mean_review_length * FACTOR
min_length = 2

contractions = pd.read_csv('./contractions.csv', index_col='Contraction')
contractions.index = contractions.index.str.lower()
contractions.Meaning = contractions.Meaning.str.lower()
contractions_dict = contractions.to_dict()['Meaning']

def preprocess_review(text):
    review = str(text)

    # Replace all occurrences
    for replace, by in replace_by:
        review = review.replace(replace, by)
    
    review = review.lower()
    review = review[:limited_text_length]

    tokens = review.split()

    reg_exp_filter_rule=re.compile("[%s]"%re.escape(string.punctuation))
    words_vector = [reg_exp_filter_rule.sub("", word) for word in tokens]

    words_vector = [word for word in words_vector if word.isalpha()]

    cleaned_review = " ".join(words_vector)

    for contraction, replacement in contractions_dict.items():
        cleaned_review = cleaned_review.replace(contraction, replacement)
    
    return cleaned_review

def process_text(texts):
    processed_texts=list()
    for text in texts:
        processed_text = preprocess_review(text)
        processed_texts.append(processed_text)
    return processed_texts

X_train_processed = process_text(texts=X_train)
X_test_processed = process_text(texts=X_test)

Embedding_dimensions = 100

# Creating Word2Vec training dataset.
Word2vec_train_data = list(map(lambda x: x.split(), X_train_processed))

word2vec_model = Word2Vec(Word2vec_train_data,
                 vector_size=Embedding_dimensions,
                 workers=8,
                 min_count=5)

print("Vocabulary Length:", len(word2vec_model.wv.key_to_index))

max_length = max([len(row.split()) for row in X_train_processed])
print("Maximum length:",max_length)

def create_and_train_tokenizer(texts):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer

def encode_reviews(tokenizer, max_length, docs):
    encoded=tokenizer.texts_to_sequences(docs)
    padded=pad_sequences(encoded, maxlen=max_length, padding="post")
    return padded

tokenizer=create_and_train_tokenizer(texts = X_train_processed)
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary size:", vocab_size)

X_train_proc = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_length)
X_test_proc  = pad_sequences(tokenizer.texts_to_sequences(X_test) , maxlen=max_length)

embedding_matrix = np.zeros((vocab_size, Embedding_dimensions))

for word, token in tokenizer.word_index.items():
    if word2vec_model.wv.__contains__(word):
        embedding_matrix[token] = word2vec_model.wv.__getitem__(word)

print("Embedding Matrix Shape:", embedding_matrix.shape)

from sklearn.metrics import confusion_matrix, classification_report

def getModel():
    embedding_layer = Embedding(input_dim = vocab_size,
                                output_dim = Embedding_dimensions,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
        Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
        Conv1D(100, 5, activation='relu'),
        GlobalMaxPool1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ],
    name="Sentiment_Model")
    return model

training_model = getModel()
training_model.summary()

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
             EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

training_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = training_model.fit(
    X_train_proc, y_train,
    batch_size=64,
    epochs=15,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)

acc,  val_acc  = history.history['accuracy'], history.history['val_accuracy']
loss, val_loss = history.history['loss'], history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('accuracy-cornell.png')

def ConfusionMatrix(y_pred, y_test):
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.savefig('confusion_matrix-cornell.png')

# Predicting on the Test dataset.
y_pred = training_model.predict(X_test_proc)

# Converting prediction to reflect the sentiment predicted.
y_pred_c = np.where(y_pred>=0.5, 1, 0)

# Printing out the Evaluation metrics.
ConfusionMatrix(y_pred_c, y_test)

# Print the evaluation metrics for the dataset.
print(classification_report(y_test, y_pred_c))

auc = roc_auc_score(y_test, y_pred)
prec = precision_score(y_test, y_pred_c)
rec = recall_score(y_test, y_pred_c)
f1 = f1_score(y_test, y_pred_c)
    
print('auc :{}'.format(auc))
print('precision :{}'.format(prec))
print('recall :{}'.format(rec))
print('f1 :{}'.format(f1))

balanced_accuracy = balanced_accuracy_score(y_test, y_pred_c)
print('balanced_accuracy :{}'.format(balanced_accuracy))

