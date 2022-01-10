# # https://drive.google.com/file/d/1qeCxZAl-h7R8kILzfVQoGM44JIY5HUpe/view?usp=sharing
# !gdown --id 1qeCxZAl-h7R8kILzfVQoGM44JIY5HUpe

q1/////


> q1





import numpy as np
!pip install --upgrade pip
!pip uninstall tfa-nightly
!pip install tensorflow-addons
import tensorflow as tf 
import tensorflow_addons as tfa

!gdown --id 104TU-gdBAtalfRyaauuKaOobhGbB52Zf
!unzip "/content/data.zip"

![](https://drive.google.com/uc?export=view&id=1_Y5cKUrCs8r9ZiVfzmgpfJJOQum_we39)

import json
f= open ("/content/ner_intent.json", "r")
data = json.loads(f.read())

labels=[]
text_all=[]
tag_all=[]
intent_all=[]
intent_unique=[]

for i in data:
  text=i
  text_s=data[i]["TEXT"]
  tag=data[i]["NERTAGS"]
  intent=data[i]["INTENTS"]
  NERVLAS=data[i]["NERVLAS"]
  intent_new=[]
  # for y in intent:
  #   intent_new.append(intent2index[y])
  for t in intent:
    if not t in intent_unique:
      intent_unique.append(t)
  tag_new=[]
  # for j in tag:
  #   tag_new.append(tags2index[j])

  text_all.append(text)
  tag_all.append(tag_new)
  intent_all.append(intent_new)
  
  for j in tag:
    if not j in labels:
      labels.append(j)
tags2index = {t:i for i,t in enumerate(labels)}
print(tags2index)
intent2index={t:i for i,t in enumerate(intent_unique)}
intent2index


labels=[]
text_all=[]
tag_all=[]
intent_all=[]
intent_unique=[]

for i in data:
  text=i
  text_s=data[i]["TEXT"]
  tag=data[i]["NERTAGS"]
  intent=data[i]["INTENTS"]
  NERVLAS=data[i]["NERVLAS"]
  intent_new=[]
  for y in intent:
    intent_new.append(intent2index[y])
  for t in intent:
    if not t in intent_unique:
      intent_unique.append(t)
  tag_new=[]
  for j in tag:
    tag_new.append(tags2index[j])

  text_all.append(text)
  tag_all.append(tag_new)
  intent_all.append(intent_new)
  
  for j in tag:
    if not j in labels:
      labels.append(j)
tags2index = {t:i for i,t in enumerate(labels)}
print(tags2index)
intent2index={t:i for i,t in enumerate(intent_unique)}
intent2index


pop=[]
for i in intent_all:
  p=[0]*27
  for j in i:
    p[j]=1
  pop.append(p)
# pop
  

sentences_train=text_all[0:1740]
seq_tags_train=tag_all[0:1740]
intent_train=intent_all[0:1740]

# intent_all
sentences_val=text_all[1740:1836]
seq_tags_val=tag_all[1740:1836]
intent_val=intent_all[1740:1836]

sentences_t=text_all[1836:]
seq_tags_test=tag_all[1836:]
intent_test=intent_all[1836:]

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences_train+sentences_val)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_t)
X_val = tokenizer.texts_to_sequences(sentences_val)

vocab_size = len(tokenizer.word_index) + 1 

sent_length = {}

for elem in X_train:
  res = len(elem)
  if res in sent_length:
    sent_length[res]+=1
  else:
    sent_length[res]=1

print(sent_length)

import matplotlib.pyplot as plt

x = [key for key in sent_length]
y = [sent_length[key] for key in sent_length]

plt.scatter(x, y)
plt.show()

max_len = 20

from keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)
X_val= pad_sequences(X_val, padding='post', maxlen=max_len)

seq_tags_train = pad_sequences(seq_tags_train,  padding='post', maxlen=max_len, value=tags2index['O'])
seq_tags_test = pad_sequences(seq_tags_test,  padding='post', maxlen=max_len, value=tags2index["O"])
seq_tags_val = pad_sequences(seq_tags_val,  padding='post', maxlen=max_len, value=tags2index["O"])

seq_tags_train = seq_tags_train.reshape(seq_tags_train.shape[0], seq_tags_train.shape[1], 1)
seq_tags_test = seq_tags_test.reshape(seq_tags_test.shape[0], seq_tags_test.shape[1], 1)
seq_tags_val = seq_tags_val.reshape(seq_tags_val.shape[0], seq_tags_test.shape[1], 1)


from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, GRU

embedding_dim = 100

model2 = Sequential()
model2.add(Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=max_len,
                           trainable=True))

model2.add(Bidirectional(GRU(300, dropout=0.2,  return_sequences=True)))
model2.add(TimeDistributed(Dense(11, activation="softmax")))


model2.summary()


# import tensorflow as tf
# print(tf.__version__)
# !python --
# !pip install keras-tuner

!pip install --upgrade pip
!pip uninstall tfa-nightly
!pip install tensorflow-addons
import tensorflow as tf 
import tensorflow_addons as tfa

# https://stackoverflow.com/questions/58565394/what-is-the-difference-between-sparse-categorical-crossentropy-and-categorical-c
# from keras.optimizers import SGD
...
# opt = SGD()
import tensorflow as tf 
import tensorflow_addons as tfa
# from tensorflow import kerasdef
import tensorflow as tf
opt=tf.keras.optimizers.Adam(learning_rate=2e-5)
# opt=SGD(lr=0.9)
model2.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=[tfa.metrics.F1Score(num_classes=11, average="micro",threshold=0.9)])

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = '/content/model_2-{epoch:03d}-{val_f1_score:03f}.ckpt'

cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    monitor = 'f1_score',
    mode = 'max',
    save_best_only=True)

batch_size = 4
import numpy as np

history2 = model2.fit(np.array(X_train), seq_tags_train, validation_data=(np.array(X_val), seq_tags_val),
                    batch_size=batch_size, epochs=1, verbose=1,callbacks=[cp_callback])

!zip -r /content/model2.zip /content/model_2-029-0.161105.ckpt

from google.colab import files
files.download("/content/model2.zip")


# https://drive.google.com/file/d/1DPRSGu54IjPp83iqmlBRztUbYJXT7TX4/view?usp=sharing
!gdown --id 1DPRSGu54IjPp83iqmlBRztUbYJXT7TX4
!unzip "/content/model2.zip"
from keras.models import load_model
model2 = load_model("/content/content/model_2-029-0.161105.ckpt")

pred = model2.predict(np.array(X_test))


pred.shape

pred = pred.reshape(pred.shape[0]*pred.shape[1], pred.shape[2])

predicted_label = [np.argmax(elem) for elem in pred]

test_label = seq_tags_test.reshape(seq_tags_test.shape[0]*seq_tags_test.shape[1])

from sklearn.metrics import classification_report
print(classification_report(predicted_label, test_label))

from sklearn.metrics import f1_score
print(f1_score(test_label, predicted_label, average='macro'))

!gdown --id 1-ay5mBPEFuzHGZpDnBJIA5e_ONAIcvjW

sentences_slot_test=[]
with open("/content/slot-test.txt", 'r') as f:

  for line in f:
    line = line.rstrip()
    sentences_slot_test.append(line)
sentences_slot_test

sentences_slot = tokenizer.texts_to_sequences(sentences_slot_test)
sentences_slot = pad_sequences(sentences_slot, padding='post', maxlen=max_len)

pred = model2.predict(np.array(sentences_slot))


def change_tag(j):
  result=[]
  for k in j:
    f=k[0]

    for i in tags2index:
      if tags2index[i]==f:
        result.append(i)
  return result



k=[]

for i in pred:
  g=[]


  for j in i:
    # print(j)
    max=0
    index_max=np.argmax(j)
    # for t in range(len(j)):
    #   # print(t)
    #   if j[t]>max:
    #     max=j[t]
    #     index_max=t
    g.append([index_max])
  k.append(g)
    #   # k.append(g)
for m,n in zip (k,sentences_slot_test):
  print(n)
  
  print(m)
  print(change_tag(m))

q1//////bert

from keras.preprocessing.sequence import pad_sequences
tag_all=pad_sequences(tag_all,  padding='post', maxlen=64, value=tags2index['O'])
tag_all = tag_all.reshape(tag_all.shape[0], tag_all.shape[1], 1)

labels=[]
for i in tag_all:
  # labels.append(i[0])
  labels.append(np.array(i))

!pip install transformers

from transformers import BertTokenizer
import numpy as np

input_ids=[]
attention_masks=[]

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for sent in text_all:
    bert_inp = bert_tokenizer.encode_plus(sent, add_special_tokens = True, max_length =64, pad_to_max_length = True, 
                                        return_attention_mask = True)
    input_ids.append(bert_inp['input_ids'])
    attention_masks.append(bert_inp['attention_mask'])

input_ids = np.asarray(input_ids)
attention_masks = np.array(attention_masks)
labels = np.array(labels)

from sklearn.model_selection import train_test_split

train_inp, val_inp, train_label, val_label, train_mask, val_mask = train_test_split(
   input_ids, labels, attention_masks, test_size=0.20, random_state=1000)

test_inp, val_inp, test_label, val_label, test_mask, val_mask = train_test_split(
   val_inp, val_label, val_mask, test_size=0.5, random_state=1000)

import tensorflow as tf
from transformers import TFBertModel, TFBertForPreTraining

SEQ_LEN = 64

bert = TFBertModel.from_pretrained('bert-base-uncased')

input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

embeddings = bert(input_ids, attention_mask=mask)[0]
# X = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1))(embeddings)
y = tf.keras.layers.Dense(len(tags2index), activation='softmax', name='outputs')(embeddings)
# y=tf.keras.layers.TimeDistributed(name="jj")(y)

bert_model_slot = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

bert_model_slot.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[tfa.metrics.F1Score(num_classes=11, average="micro",threshold=0.9)])

bert_model_slot.summary()

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = '/content/bert_model_slot-{epoch:03d}-{val_f1_score:03f}.h5'

cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    monitor = 'f1_score',
    mode = 'max',
    save_best_only=True,
    save_weights_only=True)

history = bert_model_slot.fit([train_inp, train_mask], train_label,
                       batch_size = 32,
                       epochs = 1,
                       validation_data = ([val_inp, val_mask],val_label),callbacks=[cp_callback])


saving part

# !zip -r /content/bert_model_slot.zip /content/bert_model_slot-003-0.165494.h5
# from google.colab import files
# files.download("/content/bert_model_slot.zip")


# https://drive.google.com/file/d/11qSAO9NndiUsdZfqleK_ODX43K68qtab/view?usp=sharing
!gdown --id 11qSAO9NndiUsdZfqleK_ODX43K68qtab
!unzip "/content/bert_model_slot.zip"

import tensorflow as tf
from transformers import TFBertModel, TFBertForPreTraining

SEQ_LEN = 64
bert = TFBertModel.from_pretrained('bert-base-uncased')

input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

embeddings = bert(input_ids, attention_mask=mask)[0]
# X = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1))(embeddings)
y = tf.keras.layers.Dense(len(tags2index), activation='softmax', name='outputs')(embeddings)
# y=tf.keras.layers.TimeDistributed(name="jj")(y)

bert_model_slot = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)
bert_model_slot.load_weights("/content/content/bert_model_slot-003-0.165494.h5")
bert_model_slot.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[tfa.metrics.F1Score(num_classes=11, average="micro",threshold=0.9)])

slot_pred = bert_model_slot.predict([test_inp,test_mask])




slot_pred = slot_pred.reshape(slot_pred.shape[0]*slot_pred.shape[1], slot_pred.shape[2])
predicted_label = [np.argmax(elem) for elem in slot_pred]


test_label = test_label.reshape(test_label.shape[0]*test_label.shape[1])


from sklearn.metrics import classification_report
print(classification_report(predicted_label, test_label))
# predicted_label
from sklearn.metrics import f1_score
print(f1_score(test_label, predicted_label, average='macro'))

def bert_sent(data):
  from transformers import BertTokenizer
  import numpy as np

  input_ids=[]
  attention_masks=[]

  bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  for sent in data:
      bert_inp = bert_tokenizer.encode_plus(sent, add_special_tokens = True, max_length =64, pad_to_max_length = True, 
                                          return_attention_mask = True)
      input_ids.append(bert_inp['input_ids'])
      attention_masks.append(bert_inp['attention_mask'])

  input_ids = np.asarray(input_ids)
  attention_masks = np.array(attention_masks)
  return [input_ids,attention_masks]





!gdown --id 1-ay5mBPEFuzHGZpDnBJIA5e_ONAIcvjW

sentences_slot_test=[]
with open("/content/slot-test.txt", 'r') as f:

  for line in f:
    line = line.rstrip()
    sentences_slot_test.append(line)
sentences_slot_test

pred1 = bert_model_slot.predict(bert_sent(sentences_slot_test))


k=[]

for i in pred1:
  g=[]


  for j in i:
    # print(j)
    max=0
    index_max=np.argmax(j)
    # for t in range(len(j)):
    #   # print(t)
    #   if j[t]>max:
    #     max=j[t]
    #     index_max=t
    g.append([index_max])
  k.append(g)
    #   # k.append(g)
for m,n in zip (k,sentences_slot_test):
  print(n)
  
  print(m)
  print(change_tag(m))

q2///bert

> q2

---





sentences_train=text_all[0:1740]
seq_tags_train=tag_all[0:1740]
intent_train=intent_all[0:1740]

# intent_all
sentences_val=text_all[1740:1836]
seq_tags_val=tag_all[1740:1836]
intent_val=intent_all[1740:1836]

sentences_t=text_all[1836:]
seq_tags_test=tag_all[1836:]
intent_test=intent_all[1836:]

!pip install transformers

pop=[]
for i in intent_all:
  p=[0]*27
  for j in i:
    p[j]=1
  pop.append(p)
# pop
  

import pandas as pd
import re

def clean(data):
  tokens = data.split()
  translation_table = str.maketrans('', '', "\"#$%&'()*+-/:;<=>@[\]^_`{|}~?!.,")
  tokens = [w.translate(translation_table) for w in tokens]
  tokens = [word.lower() for word in tokens]

  return ' '.join(tokens)

data = []
# labels = intent_all

# read news
# with open('/content/drive/MyDrive/dataset/SMSSpamCollection.txt') as f:
#     lines = [line.rstrip() for line in f]

for i in text_all:
  # tmp = line.split('\t')
  data.append(clean(i))
# labels=intent_all[:]
labels=[]
for i in pop:
  labels.append(np.array(i))
labels=np.array(labels)
# for i in intent_all:
#   labels.append()
  # l=[]
  # for j in i:
  #   l.append(np.array(j))
  # labels.append(l)
  # # labels.append(i[0])
  # labels.append(np.asarray(i).astype('float32'))
  

from transformers import BertTokenizer
import numpy as np

input_ids=[]
attention_masks=[]

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for sent in data:
    bert_inp = bert_tokenizer.encode_plus(sent, add_special_tokens = True, max_length =64, pad_to_max_length = True, 
                                        return_attention_mask = True)
    input_ids.append(bert_inp['input_ids'])
    attention_masks.append(bert_inp['attention_mask'])

input_ids = np.array(input_ids)
attention_masks = np.array(attention_masks)


# labels=np.array(labels)

from sklearn.model_selection import train_test_split

train_inp, val_inp, train_label, val_label, train_mask, val_mask = train_test_split(
   input_ids, labels, attention_masks, test_size=0.20, random_state=1000)

test_inp, val_inp, test_label, val_label, test_mask, val_mask = train_test_split(
   val_inp, val_label, val_mask, test_size=0.5, random_state=1000)

len(intent2index)

import tensorflow as tf
from transformers import TFBertModel, TFBertForPreTraining

SEQ_LEN = 64

bert = TFBertModel.from_pretrained('bert-base-uncased')

input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

embeddings = bert(input_ids, attention_mask=mask)[0]
X = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1))(embeddings)
y = tf.keras.layers.Dense(len(intent2index), activation='sigmoid', name='outputs')(X)

bert_model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

bert_model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=[tfa.metrics.F1Score(num_classes=11, average="micro",threshold=0.9)])


bert_model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = '/content/bert_model-{epoch:03d}-{val_f1_score:03f}.h5'

cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    monitor = 'f1_score',
    mode = 'max',
    save_best_only=True,
    save_weights_only=True)

history = bert_model.fit([train_inp, train_mask], train_label,
                       batch_size = 2,
                       epochs = 1,
                       validation_data = ([val_inp, val_mask],val_label),callbacks=[cp_callback]
                    )


# !zip -r /content/bert_model_intent.zip /content/bert_model-016-0.963671.h5

# from google.colab import files
# files.download("/content/bert_model_intent.zip")


# https://drive.google.com/file/d/1mkvSIDc0vLXFbmN5568EUlbtfY_3CyY7/view?usp=sharing
# https://drive.google.com/file/d/13xqIGSUazeUUOuO4ZaZuiiiypjIv-mpO/view?usp=sharing
!gdown --id 13xqIGSUazeUUOuO4ZaZuiiiypjIv-mpO
!unzip "/content/bert_model_intent.zip"
# from keras.models import load_model
# model = load_model("")

import tensorflow as tf
from transformers import TFBertModel, TFBertForPreTraining

SEQ_LEN = 64

bert = TFBertModel.from_pretrained('bert-base-uncased')

input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

embeddings = bert(input_ids, attention_mask=mask)[0]
X = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1))(embeddings)
y = tf.keras.layers.Dense(len(intent2index), activation='softmax', name='outputs')(X)

bert_model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)
bert_model.load_weights("/content/content/bert_model-016-0.963671.h5")
bert_model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=[tfa.metrics.F1Score(num_classes=11, average="micro",threshold=0.9)])

bert_model.summary()

intent_pred = bert_model.predict([test_inp,test_mask])



new_pred=[]
for t in intent_pred:
  k=[]
  for j in t :
    if j>0.7:
      # j=1
      k.append(1)
    if j<= 0.7:
      k.append(0)
    
  new_pred.append(np.array(k))

new_pred_=np.array(new_pred)


micro caculation on test data

# intent_pred
# predicted_label = [np.argmax(elem) for elem in intent_pred]
# # predicted_label
from sklearn.metrics import f1_score
print(f1_score(test_label, new_pred, average='micro'))
from sklearn.metrics import classification_report
print(classification_report(new_pred, test_label))
# predicted_label


!gdown --id 101uOTEhpm7t3Y4iIPpA_isGUHGKXafsA
sentences_slot_test=[]
with open("/content/intent-test.txt", 'r') as f:

  for line in f:
    line = line.rstrip()
    sentences_slot_test.append(line)
sentences_slot_test

def bert_sent(data):
  from transformers import BertTokenizer
  import numpy as np

  input_ids=[]
  attention_masks=[]

  bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  for sent in data:
      bert_inp = bert_tokenizer.encode_plus(sent, add_special_tokens = True, max_length =64, pad_to_max_length = True, 
                                          return_attention_mask = True)
      input_ids.append(bert_inp['input_ids'])
      attention_masks.append(bert_inp['attention_mask'])

  input_ids = np.asarray(input_ids)
  attention_masks = np.array(attention_masks)
  return [input_ids,attention_masks]





pred1 = bert_model.predict(bert_sent(sentences_slot_test))

def new_change(new_pred0):
  result=[]
  for i in range (len(new_pred0)):
    # for j in i:
    if new_pred0[i]==1:
      t=i
      # print(t)
      for ii in intent2index:
        if intent2index[ii]==t:
            result.append(ii)
    
  return result

j=[0]

def change(j):
  result=[]
  for k in j:
    f=k

  for i in intent2index:
    if intent2index[i]==f:
        result.append(i)
  return result
change(j)

mohasebe intent jomalat

# pred[0][0]
# k=[]

# for i in pred1:

#   index_max=np.argmax(i)

#   k.append([index_max])
new_pred=[]
for t in pred1:
  k=[]
  for j in t :
    if j>0.44:
      j=1
      k.append(1)
    j=0
    k.append(0)
  new_pred.append(k)
# print(new_pred)

for m,n in zip (new_pred,sentences_slot_test):
  print(n)
  
  print(m)
  # print(change(m))
  print(new_change(m))




q2//without bert/// with gru and i didnt considering multi label in this section

sentences_train=text_all[0:1740]
seq_tags_train=tag_all[0:1740]
intent_train=intent_all[0:1740]

# intent_all
sentences_val=text_all[1740:1836]
seq_tags_val=tag_all[1740:1836]
intent_val=intent_all[1740:1836]

sentences_t=text_all[1836:]
seq_tags_test=tag_all[1836:]
intent_test=intent_all[1836:]

import numpy as np
labels=[]
for i in intent_train:
  # labels.append(i[0])
  labels.append(np.array(i[0]))

labels_val=[]
for i in intent_val:
  # labels.append(i[0])
  labels_val.append(np.array(i[0]))

labels_test=[]
for i in intent_test:
  # labels.append(i[0])
  labels_test.append(np.array(i[0]))

labels = np.asarray(labels).astype('float32')
labels_val = np.asarray(labels_val).astype('float32')
labels_test = np.asarray(labels_test).astype('float32')

import numpy as np
from tensorflow.keras.utils import to_categorical

# labels= to_categorical(labels, num_classes=len(intent_unique))

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences_train+sentences_val)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_t)
X_val = tokenizer.texts_to_sequences(sentences_val)

vocab_size = len(tokenizer.word_index) + 1 

max_len=20

from keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)
X_val= pad_sequences(X_val, padding='post', maxlen=max_len)

vocab_size

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, GRU

embedding_dim = 100

model = Sequential()
model.add(Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=max_len,
                           trainable=True))

model.add(Bidirectional(GRU(300, dropout=0.2)))
model.add(Dense(len(intent_unique), activation="softmax"))


model.summary()

# https://stackoverflow.com/questions/58565394/what-is-the-difference-between-sparse-categorical-crossentropy-and-categorical-c
# from keras.optimizers import SGD
# ...
import tensorflow as tf 
import tensorflow_addons as tfa
# opt = SGD()
import tensorflow as tf
opt=tf.keras.optimizers.Adam(learning_rate=2e-5)
# opt=SGD(lr=0.9)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=[tfa.metrics.F1Score(num_classes=11, average="micro",threshold=0.5)])

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = '/content/model-{epoch:03d}-{val_f1_score:03f}.ckpt'

cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    monitor = 'f1_score',
    mode = 'max',
    save_best_only=True)

batch_size = 2
import numpy as np

history = model.fit(X_train, labels, validation_data=(X_val, labels_val),
                    batch_size=batch_size, epochs=50, verbose=1,callbacks=[cp_callback])

!zip -r /content/model.zip /content/model-050-0.049411.ckpt

from google.colab import files
files.download("/content/model.zip")


# https://drive.google.com/file/d/1b15nevKU6tl_TGeoTHujAXLYburnEsRG/view?usp=sharing
!gdown --id 1b15nevKU6tl_TGeoTHujAXLYburnEsRG
!unzip "/content/model.zip"
from keras.models import load_model
model = load_model("/content/content/model-050-0.049411.ckpt")

intent_pred = model.predict(np.array(X_test))


labels_test.shape

intent_pred
predicted_label = [np.argmax(elem) for elem in intent_pred]
# predicted_label
from sklearn.metrics import classification_report
print(classification_report(predicted_label, labels_test))
# predicted_label
from sklearn.metrics import f1_score
print(f1_score(labels_test, predicted_label, average='micro'))

!gdown --id 101uOTEhpm7t3Y4iIPpA_isGUHGKXafsA
# !gdown --id 1-ay5mBPEFuzHGZpDnBJIA5e_ONAIcvjW

!gdown --id 101uOTEhpm7t3Y4iIPpA_isGUHGKXafsA
sentences_slot_test=[]
with open("/content/intent-test.txt", 'r') as f:

  for line in f:
    line = line.rstrip()
    sentences_slot_test.append(line)
sentences_slot_test

# sentences_slot_test=sentences_train[0:3]

sentences_slot = tokenizer.texts_to_sequences(sentences_slot_test)
sentences_slot = pad_sequences(sentences_slot, padding='post', maxlen=max_len)

intent_pred = model.predict(np.array(sentences_slot))

intent_pred[0]

j=[0]

def change(j):
  result=[]
  for k in j:
    f=k

  for i in intent2index:
    if intent2index[i]==f:
        result.append(i)
  return result
change(j)

# pred[0][0]
k=[]

for i in intent_pred:

  index_max=np.argmax(i)

  k.append([index_max])

for m,n in zip (k,sentences_slot_test):
  print(n)
  
  print(m)
  print(change(m))


q3//a

list_item=['B-name','I-name','B-pricerange','I-pricerange','I-area','B-area','B-food','I-food']

f= open ("/content/simple-dstc2-trn.json", "r")

# Reading from file
data= json.loads(f.read())

result=[]
for s in data:
  for i in s:

    if "db_result" in list(i.keys()):
      result.append(i["db_result"])
f= open ("/content/simple-dstc2-tst.json", "r")

# Reading from file
data= json.loads(f.read())

# result=[]
for s in data:
  for i in s:

    if "db_result" in list(i.keys()):
      result.append(i["db_result"])
f= open ("/content/simple-dstc2-val.json", "r")

# Reading from file
data= json.loads(f.read())

# result=[]
for s in data:
  for i in s:

    if "db_result" in list(i.keys()):
      result.append(i["db_result"])

without_repeated_result=result[:]
for i in result:
  if i=={}:
    without_repeated_result.remove(i)
names=[]
without_same=without_repeated_result[:]
for i in without_repeated_result:
  if not i["name"] in names:
    names.append(i["name"])
  else:
    without_same.remove(i)

print(len(without_same))
nested_result_list=without_same.copy()
# print(nested_result_list)

#############################

new_result=[]
for i in nested_result_list:
  d=i
  l = ["name","food","area","pricerange","phone","addr","postcode"]
  if not "name" in list(d.keys()):
    d["name"]=" "
  if not "food" in list(d.keys()):
    d["food"]=" "
  if not "area" in list(d.keys()):
    d["area"]=" "
  if not "pricerange" in list(d.keys()):
    d["pricerange"]=" "
  if not "phone" in list(d.keys()):
    d["phone"]=" "
  if not "addr" in list(d.keys()):
    d["addr"]=" "
  if not "postcode" in list(d.keys()):
    d["postcode"]=" "
  ordered_dict_items = [(k,d[k]) for k in l]
  new_result.append(dict(ordered_dict_items))



#################################

len(new_result)


from operator import itemgetter
newlist = sorted(new_result, key=itemgetter('name')) 


q3///b

print(len(newlist))
print(newlist)
original=newlist[:]

q3///1

# !gdown --id 101uOTEhpm7t3Y4iIPpA_isGUHGKXafsA
!gdown --id 101uOTEhpm7t3Y4iIPpA_isGUHGKXafsA

sentences_slot_test=[]
with open("/content/intent-test.txt", 'r') as f:

  for line in f:
    line = line.rstrip()
    sentences_slot_test.append(line)
sentences_slot_test

# sentences_slot = tokenizer.texts_to_sequences(sentences_slot_test)
# sentences_slot = pad_sequences(sentences_slot, padding='post', maxlen=max_len)

def bert_sent(data):
  from transformers import BertTokenizer
  import numpy as np

  input_ids=[]
  attention_masks=[]

  bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  for sent in data:
      bert_inp = bert_tokenizer.encode_plus(sent, add_special_tokens = True, max_length =64, pad_to_max_length = True, 
                                          return_attention_mask = True)
      input_ids.append(bert_inp['input_ids'])
      attention_masks.append(bert_inp['attention_mask'])

  input_ids = np.asarray(input_ids)
  attention_masks = np.array(attention_masks)
  return [input_ids,attention_masks]





pred1 = bert_model_slot.predict(bert_sent(sentences_slot_test))


def change_tag(j):
  result=[]
  for k in j:
    f=k[0]

    for i in tags2index:
      if tags2index[i]==f:
        result.append(i)
  return result


q3-1 result


k=[]

for i in pred1:
  g=[]


  for j in i:
    # print(j)
    max=0
    index_max=np.argmax(j)

    g.append([index_max])
  k.append(g)
    
for m,n in zip (k,sentences_slot_test):
  print(n)
  # print(m)
  f=change_tag(m)
  print(change_tag(m))
  pp=[]
  # y=[]
  newlist=original[:]
  count=0
  c=0
  for t in f:
    # print(len(f),count)
    
    count+=1
    # y=[]
    # pp=[]
    if t in list_item:
      c+=1
      y=[]
      
      r=t.split('-')[1]
      # pp=[]
      # print(r)
      for j in newlist:
        rtf=j[r].split(" ")[0]
        if rtf==n.split()[f.index(t)]:

          y.append(j)
      newlist=y[:]
      

    if count==(len(f)-1) and c>0 :
     
      for i in newlist:

        pp.append(i["name"])
  print(len(pp))
  print("list_restaurants",pp)
  print("              " )
  print("              " )
  print("              " )




q4

def bert_sent(data):
  from transformers import BertTokenizer
  import numpy as np

  input_ids=[]
  attention_masks=[]

  bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  for sent in data:
      bert_inp = bert_tokenizer.encode_plus(sent, add_special_tokens = True, max_length =64, pad_to_max_length = True, 
                                          return_attention_mask = True)
      input_ids.append(bert_inp['input_ids'])
      attention_masks.append(bert_inp['attention_mask'])

  input_ids = np.asarray(input_ids)
  attention_masks = np.array(attention_masks)
  return [input_ids,attention_masks]





def intent(sent):
  data=[sent]
  # sentences_slot = tokenizer.texts_to_sequences(data)
  # sentences_slot = pad_sequences(sentences_slot, padding='post', maxlen=max_len)
  # predkk = model.predict(np.array(sentences_slot))
  

  pred1 = bert_model.predict(bert_sent(data))
  new_pred=[]
  for t in pred1:
    k=[]
    for j in t :
      if j>0.5:
        j=1
        k.append(1)
      j=0
      k.append(0)
    new_pred.append(k)
# print(new_pred)

# for m,n in zip (new_pred,sentences_slot_test):
#   print(n)
  
#   print(m)
  # print(change(m))
  # print(new_change(new_pred[0]))
  
  # k=[]

  # for i in pred1[0]:
  # # for i in predkk:

  #   index_max=np.argmax(i)

  #   k.append([index_max])

  # m=k
  # f=change(m[0])
  return new_change(new_pred[0])
 
intent("oh thanks")

def slot(sent):
  data=[sent]
  pred2 = bert_model_slot.predict(bert_sent(data))
  # sentences_slot = tokenizer.texts_to_sequences(data)
  # sentences_slot = pad_sequences(sentences_slot, padding='post', maxlen=max_len)
  # pred2 = model2.predict(np.array(sentences_slot))

  k=[]

  for i in pred2:
    g=[]


    for j in i:
      
      
      index_max=np.argmax(j)

      g.append([index_max])
    k.append(g)
  # print(k)
  m=k
  f=change_tag(m[0])
  return f
# slot("asian oriental food")
# slot(" how about west is there any there")


# newlist=original[:]
# i_i =input()
# if i_i in ["hi","Hello","Hi","hello"]:
#   print("Hello, welcome to the Cambridge restaurant system.","\n"
#   "You can ask for restaurants by area, price range or food type. How may I help you?")
#   selected_slot={}
#   i_i =input()
#   r=intent(i_i)
#   # print(r)
#   for q in r:
#     # print(q)
#     if q in ["inform_area","inform_food","inform_pricerange"]:
#       f=slot(i_i)
#       # print(f)
#       if "B-pricerange" in f:
#         selected_slot["pricerange"]=i_i.split()[f.index("B-pricerange")]
#         print("What kind of food would you like?")
#         t_t=input()
#         if "B-food" in f:
#           selected_slot["food"]=t_t.split()[h.index("B-food")]


#         # print(t_t)
#         h=slot(t_t)
#         list_r=[]
#         for t in newlist:
#           if t["pricerange"]==i_i.split()[f.index("B-pricerange")] and t["food"].split(" ")[0]==t_t.split()[h.index("B-food")]:
#             ff=t
#             list_r.append(t)
#         first_suggestion=list_r[0]["name"]
#         print(first_suggestion,"serves",list_r[0]["food"],"food in the",list_r[0]["pricerange"],"price range.")
#         f_f=input()
#         if "B-area" in slot(f_f)or f_f=="address":
#           if list_r[0]["area"]==" ":
#             print("this restarant's area hasnt entered")
#           print(list_r[0]["name"] ,"is in ",list_r[0]["area"])
#           g_g=input()
#           if "B-area" in slot(g_g):
#             list_r=[]
#             for t in newlist:
#               if t["area"]==g_g.split()[f.index("B-area")] and t["pricerange"]==i_i.split()[f.index("B-pricerange")] and t["food"].split(" ")[0]==t_t.split()[h.index("B-food")]:
#                 ff=t
#                 list_r.append(t)
#             if len(list_r)==0:
#               print("Sorry there is no ",list_r[0]["pricerange"], "restaurant in the ",g_g.split()[f.index("B-area")]," of town serving ",t_t.split()[h.index("B-food")]," food.")
#             first_suggestion=list_r[0]["name"]
#             print(first_suggestion)
#             u_u=input()
#             print("bye")




#         # print(len(list_r),"restaurant found")
#         # print()
#         # print("this is their info","\n",list_r)
#     # if "B-area" in f:
#     #   print("What kind of food would you like?")
#     #   r_r=input()
#     #   g=slot(r_r)
#     #   list_r=[]
#     #   for t in newlist:
#     #     if t["area"]==i_i.split()[f.index("B-area")] and t["food"]==r_r.split()[g.index("B-food")]:
#     #       ff=t
#     #       list_r.append(t)
#     #   print(len(list_r),"restaurant found")
#     #   # print()
#     #   print("this is their info","\n",list_r)
# else:
#   r=intent(i_i)
#   if r in ["inform_area"]:
#     f=slot(i_i)
#     if "B-area" in f:
#       print("What kind of food would you like?")
#       r_r=input()
#       g=slot(r_r)
#       list_r=[]
#       for t in newlist:
#         if t["area"]==i_i.split()[f.index("B-area")] and t["food"]==r_r.split()[g.index("B-food")]:
#           ff=t
#           list_r.append(t)
#       print(len(list_r),"restaurant found")
#       # print()
#       print("this is their info","\n",list_r)

dialoge cell


def intent(sent):
  data=[sent]
  # sentences_slot = tokenizer.texts_to_sequences(data)
  # sentences_slot = pad_sequences(sentences_slot, padding='post', maxlen=max_len)
  # predkk = model.predict(np.array(sentences_slot))
  

  pred1 = bert_model.predict(bert_sent(data))
  new_pred=[]
  for t in pred1:
    k=[]
    for j in t :
      if j>0.5:
        j=1
        k.append(1)
      j=0
      k.append(0)
    new_pred.append(k)
# print(new_pred)


  return new_change(new_pred[0])
 

newlist=original[:]
u=[]
i_i=input()
intent=intent(i_i)
# print(intent)
s=slot(i_i)
slot_dict=dict(zip(s,i_i.split(" ")))
list_slot=list(slot_dict.keys())
# print(slot_dict)
if intent==['hello']:
  print("Hello, welcome to the Cambridge restaurant system.","\n"
  "You can ask for restaurants by area, price range or food type. How may I help you?")
  def intent(sent):
    data=[sent]
    # sentences_slot = tokenizer.texts_to_sequences(data)
    # sentences_slot = pad_sequences(sentences_slot, padding='post', maxlen=max_len)
    # predkk = model.predict(np.array(sentences_slot))
    

    pred1 = bert_model.predict(bert_sent(data))
    new_pred=[]
    for t in pred1:
      k=[]
      for j in t :
        if j>0.5:
          j=1
          k.append(1)
        j=0
        k.append(0)
      new_pred.append(k)
  # print(new_pred)


    return new_change(new_pred[0])

  i_i=input()
  intent=intent(i_i)
  print(intent)
  s=slot(i_i)
  slot_dict=dict(zip(s,i_i.split(" ")))
  list_slot=list(slot_dict.keys())
  # print(slot_dict)
  if intent==['request_phone']:
    # print("22222222")
    ff=[]
    
    for i in list_slot:
      if i in list_item:
        # print(i)
        for t in newlist:
          u=[]
          # print(i.split("-")[1])
          if t[i.split("-")[1]]==slot_dict[i]:
            ff.append(t)
            u.append(t)
        newlist=ff
    # print(u)
  if intent==['inform_pricerange']:
    if not "B-food" in list_slot:
      print("What kind of food would you like?")
      n_n=input()
      # intent=intent(n_n)
      # print(intent)
      ss=slot(n_n)
      
      slot_dict_=dict(zip(ss,n_n.split(" ")))
      slot_dict.update(slot_dict_)
      list_slot=list(slot_dict.keys())
      # print(slot_dict)

      ff=[]
      # print(list_slot)  
      for i in list_slot:
        # print("i",i)
        if i in list_item:
          # print(i)
          u=[]
          for t in newlist:
            
            # print(i.split("-")[1])
            if t[i.split("-")[1]]==slot_dict[i]:
              ff.append(t)
              u.append(t)
          # print(ff)
          
          newlist=ff[:]
    first_suggestion=u[0]["name"]
    print(first_suggestion,"serves",u[0]["food"],"food in the",u[0]["pricerange"],"price range.")
    def intent(sent):
      data=[sent]
      # sentences_slot = tokenizer.texts_to_sequences(data)
      # sentences_slot = pad_sequences(sentences_slot, padding='post', maxlen=max_len)
      # predkk = model.predict(np.array(sentences_slot))
      

      pred1 = bert_model.predict(bert_sent(data))
      new_pred=[]
      for t in pred1:
        k=[]
        for j in t :
          if j>0.5:
            j=1
            k.append(1)
          j=0
          k.append(0)
        new_pred.append(k)
    # print(new_pred)


      return new_change(new_pred[0])
    j_j=input()
    if intent(j_j)==['request_addr']:
      print("Sure, ",first_suggestion," is on ",u[0]["area"])
      a_a=input()
      ss=slot(a_a)
      
      slot_dict_=dict(zip(ss,a_a.split(" ")))
      slot_dict.update(slot_dict_)
      list_slot=list(slot_dict.keys())
      # print(slot_dict)

      if intent(a_a)==['inform_area']:
        pos=list_slot.index("B-area")
        pof=list_slot.index("B-food")
        pow=list_slot.index("B-pricerange")
        # print(u)
        # print(pos)
        for e in u:
          if e[list_slot[pos].split("-")[1]]==slot_dict[list_slot[pos]]:
            print(e)
          else:
            not_exist=True
        if not_exist==True:
          print("Sorry there is no ",slot_dict[list_slot[pow]]," restaurant in the ",slot_dict[list_slot[pos]]," of town serving ",slot_dict[list_slot[pof]]," food.")

        w_w=input()
        if intent(w_w)==['thankyou']:
          print("You are welcome!")


      

    if intent(j_j)==['reqalts']:
      if len(u)>1:
        print("another suggestion:",u[1])
if intent==['request_phone']:
  
  ff=[]
  
  for i in list_slot:
    if i in list_item:
      # print(i)
      for t in newlist:
        u=[]
        # print(i.split("-")[1])
        if t[i.split("-")[1]]==slot_dict[i]:
          ff.append(t)
          u.append(t)
      newlist=ff
  print(u)
if intent==['inform_pricerange']:
  if not "B-food" in list_slot:
    print("What kind of food would you like?")
    n_n=input()
    # intent=intent(n_n)
    print(intent)
    ss=slot(n_n)
    
    slot_dict_=dict(zip(ss,n_n.split(" ")))
    slot_dict.update(slot_dict_)
    list_slot=list(slot_dict.keys())
    print(slot_dict)

    ff=[]
    print(list_slot)  
    for i in list_slot:
      # print("i",i)
      if i in list_item:
        # print(i)
        u=[]
        for t in newlist:
          
          # print(i.split("-")[1])
          if t[i.split("-")[1]]==slot_dict[i]:
            ff.append(t)
            u.append(t)
        # print(ff)
        
        newlist=ff[:]
  first_suggestion=u[0]["name"]
  print(first_suggestion,"serves",u[0]["food"],"food in the",u[0]["pricerange"],"price range.")
  j_j=input()
  if intent(j_j)==['reqalts']:
    if len(u)>1:
      print("another suggestion:",u[1])
    


# print(u)
    # g_g.split()[f.index("B-area")] and t["pricerange"]==i_i.split()[f.index("B-pricerange")] and t["food"].split(" ")[0]==t_t.split()[h.index("B-food")]:
    #   ff=t
    #   list_r.append(t)


che# مثال هایی با جواب بالا:


# phone number of a cheap restaurant

# Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
# /usr/local/lib/python3.7/dist-packages/transformers/tokenization_utils_base.py:2217: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
#   FutureWarning,

# ['request_phone']

# Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.

# {'O': 'restaurant', 'B-pricerange': 'cheap'}
# [{'name': 'zizzi cambridge', 'food': 'italian', 'area': 'centre', 'pricerange': 'cheap', 'phone': '01223 365599', 'addr': '47-53 regent street', 'post

