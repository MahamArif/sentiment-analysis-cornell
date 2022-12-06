import tarfile
import os
import string, re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import precision_score, roc_auc_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report

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

X_train = process_text(texts=X_train)
X_test = process_text(texts=X_test)

m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(m_url, trainable=True)

import tokenization

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=128):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    clf_output = sequence_output[:, 0, :]
    
    lay = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    lay = tf.keras.layers.Dense(32, activation='relu')(lay)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    lay = tf.keras.layers.Dense(16, activation='relu')(lay)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(lay)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

print("Encoding data ...")
max_len = 250
train_input = bert_encode(X_train, tokenizer, max_len=max_len)
test_input = bert_encode(X_test, tokenizer, max_len=max_len)
train_labels = y_train

model = build_model(bert_layer, max_len=max_len)
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=5,
    callbacks=[checkpoint, earlystopping],
    batch_size=16,
    verbose=1
)

# Predicting on the Test dataset.
y_pred = model.predict(test_input)

# Converting prediction to reflect the sentiment predicted.
y_pred_c = np.where(y_pred>=0.5, 1, 0)

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

# Compute and plot the Confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred_c)

print(cf_matrix)

categories  = ['Negative','Positive']
group_names = ['True Neg','False Pos', 'False Neg','True Pos']
group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

print (group_percentages)
