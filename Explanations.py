#!/usr/bin/env python
# coding: utf-8

# # Seq2Seq Machine Translation with Attention

# ## 1. Preprocessing
# 
# Dataset link: https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
 
# In[1]:


# Check the size of both datasets
with open("./data/unprocessed/europarl-v7.de-en.de") as file:
    ger = [line.rstrip() for line in file]
with open("./data/unprocessed/europarl-v7.de-en.en") as file:
    eng = [line.rstrip() for line in file]

print(len(eng))
print(len(ger))


# In[3]:


# Create small trainig set to test computations 
with open("./data/processed/train.csv", "r") as file:
        with open("./data/processed/train_mini.csv", "w") as new_file:
            i = 0
            for line in file:
                if i < 50000:
                    new_file.write(line)
                    
                i+=1
print("File successfully created.")


# In[5]:


# Load spacy data
# !python -m spacy download de_core_news_sm
# !python -m spacy download en_core_web_sm


# In[6]:


from seq2seq_attention.preprocess import get_parallel_csv

# Take ">" as seperator since it is not included in the text - unique to seperate eng-ger pairs. 
# Remove any ">" from text pairs.
get_parallel_csv(path_1="./data/unprocessed/europarl-v7.de-en.de", path_2="./data/unprocessed/europarl-v7.de-en.en", new_file_path="./data/processed/en_ger_full.csv", delimiter=">")


# In[40]:


# Remove sentences with lower number of words
from seq2seq_attention.preprocess import remove_sentences
remove_sentences(data_dir="./data/processed/en_ger_full.csv", min_length=4, max_length=30, delimiter=">", new_file_path="./data/processed/en_ger_full_removed_sent_len.csv")


# In[41]:


from seq2seq_attention.preprocess import train_test_split
train_test_split(file_path="./data/processed/en_ger_full_removed_sent_len.csv", sep=">", random_seed=118, dir="./data/processed")


# In[43]:


# Check files
import pandas as pd
train = pd.read_csv("./data/processed/train.csv", header=None, sep=">", names=["ger", "eng"])
val = pd.read_csv("./data/processed/val.csv", header=None, sep=">", names=["ger", "eng"])
test = pd.read_csv("./data/processed/test.csv", header=None,sep=">", names=["ger", "eng"])


# In[44]:


train_len = len(train)
val_len = len(val)
test_len = len(test)
print(f"Train: {train_len}, Val: {val_len}, Test: {test_len}")
print(f"Total: {train_len+val_len+test_len}")
print(f"Total + Removed: {train_len+val_len+test_len+11289+824619}")


# In[11]:


train.head()


# In[12]:


val.head()


# In[13]:


test.head()


# ## 2. Initialize Dataloaders

# In[1]:


from seq2seq_attention.build_dataloaders import build_fields, build_bucket_iterator, get_datasets, build_vocab
BATCH_SIZE = 100
DEVICE = "cpu"

src_field, trg_field = build_fields()
train_set, val_set, test_set = get_datasets(train_path="./data/processed/train.csv", 
                                            val_path="./data/processed/val.csv", 
                                            test_path="./data/processed/test.csv", 
                                            src_field=src_field, 
                                            trg_field=trg_field)
build_vocab(src_field=src_field, trg_field=trg_field, train_set=train_set, min_freq=2, max_vocab_size=32000)


# In[2]:


# Check vocabulary 
len(src_field.vocab)


# In[15]:


train_loader = build_bucket_iterator(dataset=train_set, batch_size=BATCH_SIZE, device=DEVICE)
val_loader = build_bucket_iterator(dataset=val_set, batch_size=BATCH_SIZE, device=DEVICE)
test_loader = build_bucket_iterator(dataset=test_set, batch_size=BATCH_SIZE, device=DEVICE)


# In[16]:


# Retrieve sample batch
iterator = iter(train_loader)


# In[17]:


example = next(iterator)
src_batch = example.src
trg_batch = example.trg
print(src_batch[0].shape, src_batch[1].shape)
print(trg_batch[0].shape, trg_batch[1].shape)


# In[18]:


print(src_batch[1])


# In[19]:


print(trg_batch[1])


# In[21]:


# itos is list of token strings with their idx 
for j in range(5):
    src = ""
    for i in src_batch[0][j]:
       src = " ".join([src,  src_field.vocab.itos[i]])
    print(src)
    trg = ""
    for i in trg_batch[j]:
        trg = " ".join([trg, trg_field.vocab.itos[i]])
    print(trg)
    print()
# The second element in the tuple is the real length that we pass to the packed_seq!


# ## 3. Evaluation

# ### 3.1 Validation loss curves for hyperparameter search

# In[2]:


import pandas as pd 

val_loss = pd.read_csv("./experiments/val_loss_clean")
val_loss.head()


# In[28]:


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(9,6))

val_loss.drop("Uniform-Attention", axis=1).iloc[:45,:].plot(ax=plt.gca())
plt.ylabel("Validation Loss")
plt.xlabel("step")
plt.xticks(np.arange(0,46,5))
plt.savefig(fname=f"report/Val_loss", dpi=150, bbox_inches='tight')
plt.show()


# In[32]:


plt.figure(figsize=(9,6))

plt.plot(val_loss["Uniform-Attention"].iloc[:45], c="C9", label="Uniform-Attention")
plt.plot(val_loss["Exp_4"].iloc[:45], c="C3", label="Experiment 4")
plt.ylabel("Validation Loss")
plt.xlabel("step")
plt.xticks(np.arange(0,46,5))
plt.legend()
plt.savefig(fname=f"report/Val_loss_uniform", dpi=150, bbox_inches='tight')
plt.show()


# ### 3.2 Test Cross Entropy Loss for learned attention vs. uniform attention

# In[2]:


from seq2seq_attention.build_dataloaders import build_fields, get_datasets, build_vocab

src, trg = build_fields()
src_field, trg_field = build_fields()
train_set, val_set, test_set = get_datasets(train_path="./data/processed/train.csv", 
                                            val_path="./data/processed/val.csv", 
                                            test_path="./data/processed/test.csv", 
                                            src_field=src_field, 
                                            trg_field=trg_field)

max_vocab_size = 8000
min_freq = 2
build_vocab(src_field=src_field, trg_field=trg_field, train_set=train_set, min_freq=min_freq, max_vocab_size=max_vocab_size)


# In[5]:


from seq2seq_attention.model import Seq2Seq_With_Attention
import torch 

lr = 1e-4
batch_size = 128
epochs = 15
enc_emb_dim = 256
hidden_dim_enc = 512
hidden_dim_dec = 512
num_layers_enc = 1
num_layers_dec = 1
emb_dim_trg = 256
device = "cuda"
teacher_forcing = 0.5
dropout=0


# Init from file
best_model_vals = torch.load("./experiments/Experiment-4/best_model_exp4.pt", map_location="cuda")
uniform_model_vals = torch.load("./experiments/Uniform-Attention/best_model.pt", map_location="cuda")

model = Seq2Seq_With_Attention(
        lr=lr,
        enc_vocab_size=len(src_field.vocab),
        vocab_size_trg=len(trg_field.vocab),
        enc_emb_dim=enc_emb_dim,
        hidden_dim_enc=hidden_dim_enc,
        hidden_dim_dec=hidden_dim_dec,
        padding_idx=src_field.vocab.stoi["<pad>"],
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        emb_dim_trg=emb_dim_trg,
        trg_pad_idx=trg_field.vocab.stoi["<pad>"],
        device=device,
        seq_beginning_token_idx=trg_field.vocab.stoi[trg_field.init_token],
        dropout=dropout,
        train_attention=True
    )


uniform_model = Seq2Seq_With_Attention(
        lr=lr,
        enc_vocab_size=len(src_field.vocab),
        vocab_size_trg=len(trg_field.vocab),
        enc_emb_dim=enc_emb_dim,
        hidden_dim_enc=hidden_dim_enc,
        hidden_dim_dec=hidden_dim_dec,
        padding_idx=src_field.vocab.stoi["<pad>"],
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        emb_dim_trg=emb_dim_trg,
        trg_pad_idx=trg_field.vocab.stoi["<pad>"],
        device=device,
        seq_beginning_token_idx=trg_field.vocab.stoi[trg_field.init_token],
        dropout=dropout,
        train_attention=False
    )

model.seq2seq.load_state_dict(best_model_vals["model_state_dict"])
uniform_model.seq2seq.load_state_dict(uniform_model_vals["model_state_dict"])


# In[7]:


from seq2seq_attention.evaluate import evaluate
from seq2seq_attention.build_dataloaders import build_bucket_iterator

# Dataloaders
test_loader = build_bucket_iterator(test_set, batch_size=256, device="cuda")

model.send_to_device()
uniform_model.send_to_device()

# Test loss
eval_loss_exp4 = evaluate(model=model, eval_loader=test_loader)
eval_loss_uniform = evaluate(model=uniform_model, eval_loader=test_loader)

print(f"Test Loss with learned attention: {eval_loss_exp4}")
print(f"Test Loss with uniform attention: {eval_loss_uniform}")


# ### 3.3 BLEU-Score comparison
# 
# The dataset "test_all_trans.csv" is the test dataset with added translations based on 
# experiment 4 and uniform attention. This needed to be done as the university computers have a good GPU for tranlation, but to compute BLEU, CPU pwer is needed and these computers have a very weak CPU which is why the tranlations needed to be manually transmitted to my local Mac-Book with better CPU.

# In[33]:


import pandas as pd


csv_reader_params = {"delimiter": ">", "skipinitialspace": True}
test_trans = pd.read_csv("./data/translation/test_all_trans.csv", **csv_reader_params)


# In[34]:


from seq2seq_attention.bleu import get_bleu_dataset

# Bleu test score for experiment 4
get_bleu_dataset(dataset=test_trans, trg_field=trg_field, trans_col="trans_exp4", parallel=True)


# In[35]:


# Bleu test score for uniform attention

get_bleu_dataset(dataset=test_trans, trg_field=trg_field, trans_col="trans_uniform", parallel=True)


# In[45]:


# Further analysis

from seq2seq_attention.bleu import get_bleu_col

test_trans["bleu_exp4"] = get_bleu_col(dataset=test_trans, trg_field=trg_field, trans_col="trans_exp4")
test_trans["bleu_uniform"] = get_bleu_col(dataset=test_trans, trg_field=trg_field, trans_col="trans_uniform")


# In[59]:


# Add sentence length column for src sentence
test_trans["src_len"] = test_trans.ger.apply(lambda sent: len(src_field.tokenize(sent)))
test_trans["trg_len"] = test_trans.eng.apply(lambda sent: len(trg_field.tokenize(sent)))


# In[79]:


test_trans.src_len.plot(kind="hist", bins=25)


# In[84]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(data=test_trans[test_trans.src_len<10], x="bleu_exp4", label="small")
sns.kdeplot(data=test_trans[(test_trans.src_len>=10)&(test_trans.trg_len<20)], x="bleu_exp4", label="med")
sns.kdeplot(data=test_trans[test_trans.src_len>=20], x="bleu_exp4", label="large")
plt.legend()


# In[150]:


# See sentences on which attention based system performs better than uniform attention with higher bleu scores
good = test_trans[test_trans.bleu_exp4>70]
good = good[good.bleu_exp4>good.bleu_uniform]
for i in range(10,15):
    print(good.ger.iloc[i])
    print(good.trans_exp4.iloc[i])
    print("")


# ## 4. Visualize Attention

# In[36]:


from seq2seq_attention.visualize_attention import get_attention_frames
from seq2seq_attention.visualize_attention import plot_attention
import matplotlib.pyplot as plt
from seq2seq_attention.visualize_attention import get_attention_frames

# Some examples that worked particularly well 
examples = ["Dies sind alles wichtige Bereiche, die wir in unserem eigenen Interesse weiterentwickeln müssen.",
            "Natürlich sind wir mit der gegenwärtigen Situation in China nicht zufrieden.",
            "Wir können die gegenwärtige Situation in Afghanistan nicht hinnehmen.",
            "Ich bin mit den Grundsätzen dieses Berichts einverstanden. "]
trans, frames = get_attention_frames(sentences=examples, model=model, src_field=src_field, trg_field=trg_field)
trans


# In[38]:


sent = "Dies sind alles wichtige Bereiche, die wir in unserem eigenen Interesse weiterentwickeln müssen. "
examples = [sent]
trans, frames = get_attention_frames(sentences=examples, model=model, src_field=src_field, trg_field=trg_field)
frame = frames[0]
plt.figure(figsize=(8,6.5))
plot = plot_attention(frame)
plt.tight_layout()
plt.savefig(fname="report/NewAttention1", dpi=150, bbox_inches='tight')


# In[39]:


sent = "Natürlich sind wir mit der gegenwärtigen Situation in China nicht zufrieden. "
examples = [sent]
trans, frames = get_attention_frames(sentences=examples, model=model, src_field=src_field, trg_field=trg_field)
frame = frames[0]
plt.figure(figsize=(8,6.5))
plot = plot_attention(frame)
plt.tight_layout()
plt.savefig(fname="report/NewAttention2", dpi=150, bbox_inches='tight')


# In[40]:


sent = "Wir können die gegenwärtige Situation in Afghanistan nicht hinnehmen. "
examples = [sent]
trans, frames = get_attention_frames(sentences=examples, model=model, src_field=src_field, trg_field=trg_field)
frame = frames[0]
plt.figure(figsize=(8,6.5))
plot = plot_attention(frame)
plt.tight_layout()
plt.savefig(fname="report/NewAttention3", dpi=150, bbox_inches='tight')


# In[41]:


sent = "Ich bin mit den Grundsätzen dieses Berichts einverstanden. "
examples = [sent]
trans, frames = get_attention_frames(sentences=examples, model=model, src_field=src_field, trg_field=trg_field)
frame = frames[0]
plt.figure(figsize=(8,6.5))
plot = plot_attention(frame)
plt.tight_layout()
plt.savefig(fname="report/NewAttention4", dpi=150, bbox_inches='tight')


# ## Appendix

# In[ ]:


from seq2seq_attention.translate import translate_sentence_without

all_translations = []
for i, sent in enumerate(test.ger.values):
    trans = translate_sentence_without(sent,
                       seq2seq_model=model.seq2seq,
                       src_field=src_field,
                bos=src_field.init_token,
                eos=src_field.eos_token,
                eos_idx=src_field.vocab.stoi[src_field.eos_token],
                trg_field=trg_field,
                max_len=30,
            )
    all_translations.append(trans)
    
    if not i%10000:
        print(f"{i/len(test)*100:.2f}% done.")


# In[ ]:


test["trans_uniform"] = all_translations


# In[ ]:


test.to_csv("./data/translation/all_trans.csv", sep=">", header=True, index=False)

