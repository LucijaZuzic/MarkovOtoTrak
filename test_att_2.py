import pandas as pd
import os
import numpy as np
from pytorch_utilities import get_XY 
from utilities import load_object, save_object
import numpy as np
from pytorch_utilities import print_predictions
import torch
from tqdm import tqdm
from time import time
from seq2seq_attention.evaluate import evaluate
from seq2seq_attention.model import Seq2Seq_With_Attention
from seq2seq_attention.build_dataloaders import (
    build_fields,
    build_bucket_iterator,
    get_datasets,
    build_vocab,
)
from seq2seq_attention.translate import translate_sentence

BATCH_SIZE = 256
DEVICE = "cuda"
LR = 5e-4 
EPOCHS = 20
MAX_VOCAB_SIZE = 8000
MIN_FREQ = 2
ENC_EMB_DIM = 256
HIDDEN_DIM_ENC = 512
HIDDEN_DIM_DEC = 512
NUM_LAYERS_ENC = 1
NUM_LAYERS_DEC = 1
EMB_DIM_TRG = 256 
TEACHER_FORCING = 0.8
PROGRESS_BAR = False
USE_WANDB = True
DROPOUT = 0
TRAIN_ATTENTION = False
progress_bar=False
disable_pro_bar = not progress_bar

def my_token(xv, yv, name_file):
    
    strpr = "x>y\n"
    for ix1 in range(len(xv)): 
        strpr += str(xv[ix1]) + ">" + str(yv[ix1]) + "\n"

    file_processed = open(name_file, "w")
    file_processed.write(strpr.replace("[", "").replace("]", "").replace(".", "a").replace(",", "a"))
    file_processed.close()

num_props = 1
 
ws_range = [1]
resave = False
if resave:
    for filename in os.listdir("actual_train"):

        varname = filename.replace("actual_train_", "")

        file_object_train = load_object("actual_train/actual_train_" + varname)
        file_object_val = load_object("actual_val/actual_val_" + varname)
        file_object_test = load_object("actual/actual_" + varname)

        for ws_use in ws_range:
            
            x_train_all = []
            y_train_all = []

            for k in file_object_train:

                x_train_part, y_train_part = get_XY(file_object_train[k], ws_use)
                
                for ix in range(len(x_train_part)):
                    x_train_all.append(x_train_part[ix]) 
                    y_train_all.append(y_train_part[ix])

            x_train_all = np.array(x_train_all)
            y_train_all = np.array(y_train_all)
            
            x_test_all = []
            y_test_all = []

            for k in file_object_test:

                x_test_part, y_test_part = get_XY(file_object_test[k], ws_use)
                
                for ix in range(len(x_test_part)):
                    x_test_all.append(x_test_part[ix]) 
                    y_test_all.append(y_test_part[ix])

            x_test_all = np.array(x_test_all)
            y_test_all = np.array(y_test_all)
            
            x_val_all = []
            y_val_all = []

            for k in file_object_val:

                x_val_part, y_val_part = get_XY(file_object_val[k], ws_use)
                
                for ix in range(len(x_val_part)):
                    x_val_all.append(x_val_part[ix]) 
                    y_val_all.append(y_val_part[ix])

            x_val_all = np.array(x_val_all)
            y_val_all = np.array(y_val_all)

            print(np.shape(x_train_all))

            if not os.path.isdir("tokenized_data/" + varname):
                os.makedirs("tokenized_data/" + varname)

            my_token(x_train_all, y_train_all, "tokenized_data/" + varname + "/" + varname + "_train_" + str(ws_use) + ".csv")
            my_token(x_val_all, y_val_all, "tokenized_data/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv")
            my_token(x_test_all, y_test_all, "tokenized_data/" + varname + "/" + varname + "_test_" + str(ws_use) + ".csv")
        
for filename in os.listdir("actual_train"):
    for ws_use in ws_range:
        varname = filename.replace("actual_train_", "")

        src_field, trg_field = build_fields()
        train_set, val_set, test_set = get_datasets(train_path="tokenized_data/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv", 
                                                    val_path="tokenized_data/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv", 
                                                    test_path="tokenized_data/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv", 
                                                    src_field=src_field, 
                                                    trg_field=trg_field)
        build_vocab(src_field=src_field, trg_field=trg_field, train_set=train_set, min_freq=MIN_FREQ, max_vocab_size=MAX_VOCAB_SIZE)
        # Check vocabulary 
        print(varname, len(src_field.vocab), len(trg_field.vocab))

        train_loader = build_bucket_iterator(dataset=train_set, batch_size=BATCH_SIZE, device=DEVICE)
        val_loader = build_bucket_iterator(dataset=val_set, batch_size=BATCH_SIZE, device=DEVICE)
        test_loader = build_bucket_iterator(dataset=test_set, batch_size=BATCH_SIZE, device=DEVICE)

        # Safe number of batches in train loader and eval points
        perc = 0.25
        n_batches_train = len(train_loader)
        eval_points = [
            round(i * perc * n_batches_train) - 1 for i in range(1, round(1 / perc))
        ]
        eval_points.append(n_batches_train - 1)

        # Get padding/<sos> idxs
        src_pad_idx = src_field.vocab.stoi["<pad>"]
        trg_pad_idx = trg_field.vocab.stoi["<pad>"]
        seq_beginning_token_idx = src_field.vocab.stoi["<sos>"]
        assert src_field.vocab.stoi["<sos>"] == trg_field.vocab.stoi["<sos>"]

        # Init model wrapper class
        model = Seq2Seq_With_Attention(
            lr=LR,
            enc_vocab_size=len(src_field.vocab),
            vocab_size_trg=len(trg_field.vocab),
            enc_emb_dim=ENC_EMB_DIM,
            hidden_dim_enc=HIDDEN_DIM_ENC,
            hidden_dim_dec=HIDDEN_DIM_DEC,
            dropout=DROPOUT,
            padding_idx=src_pad_idx,
            num_layers_enc=NUM_LAYERS_ENC,
            num_layers_dec=NUM_LAYERS_DEC,
            emb_dim_trg=EMB_DIM_TRG,
            trg_pad_idx=trg_pad_idx,
            device=DEVICE,
            seq_beginning_token_idx=seq_beginning_token_idx,
            train_attention=TRAIN_ATTENTION,
        )

        # Send model to device
        model.send_to_device()

        train_losses = []
        val_losses = []

        for epoch in range(EPOCHS):

            now = time()

            # Init loss stats for epoch
            train_loss = 0

            n_batches_since_eval = 0

            for n_batch, train_batch in enumerate(
                tqdm(
                    train_loader,
                    desc=f"Epoch {epoch}",
                    unit="batch",
                    disable=disable_pro_bar,
                )
            ):

                model.seq2seq.train()

                # Take one gradient step
                train_loss += model.train_step(
                    src_batch=train_batch.src[0],
                    trg_batch=train_batch.trg,
                    src_lens=train_batch.src[1],
                    teacher_forcing=TEACHER_FORCING,
                )

                n_batches_since_eval += 1

                # Calculate and safe train/eval losses at 25% of epoch
                if n_batch in eval_points:
    
                    now_eval = time()

                    # Evaluate
                    eval_loss = evaluate(model=model, eval_loader=val_loader)

                    print(f"Evaluation time: {(time()-now_eval)/60:.2f} minutes.")

                    # Save mean train/val loss
                    train_losses.append(train_loss / n_batches_since_eval)
                    val_losses.append(eval_loss)

                    # Set counter to 0 again
                    n_batches_since_eval = 0
                    train_loss = 0

                    print(
                        f"Epoch {epoch} [{round(n_batch*100/n_batches_train)}%]: Train loss [{train_losses[-1]}]   |  Val loss [{eval_loss}]\n"
                    )
                    print("##########################################\n")

            print(f"Epoch Training time: {(time()-now)/60:.2f} minutes.")

        model_name = "GRU_Att"
        if not os.path.isdir("train_attention2/" + varname + "/models/" + model_name):
            os.makedirs("train_attention2/" + varname + "/models/" + model_name)

        if not os.path.isdir("train_attention2/" + varname + "/predictions/train/" + model_name):
            os.makedirs("train_attention2/" + varname + "/predictions/train/" + model_name)
    
        if not os.path.isdir("train_attention2/" + varname + "/predictions/test/" + model_name):
            os.makedirs("train_attention2/" + varname + "/predictions/test/" + model_name)
    
        if not os.path.isdir("train_attention2/" + varname + "/predictions/val/" + model_name):
            os.makedirs("train_attention2/" + varname + "/predictions/val/" + model_name)
        
        save_object("train_attention2/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_train_losses", train_losses)

        save_object("train_attention2/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_val_losses", val_losses)

        model.seq2seq.eval()
        torch.save(model.seq2seq.state_dict(), "train_attention2/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + ".pth")

        with torch.no_grad():

            y_train_all = []
            predict_train_all = []
            pd_train = pd.read_csv("tokenized_data/" + varname + "/" + varname + "_train_" + str(ws_use) + ".csv", sep = ">")
            for i, ex in enumerate(list(pd_train["x"])):
                translation, _, _, _ = translate_sentence(
                    sentence=str(ex),
                    seq2seq_model=model.seq2seq,
                    src_field=src_field,
                    bos=src_field.init_token,
                    eos=src_field.eos_token,
                    eos_idx=src_field.vocab.stoi[src_field.eos_token],
                    trg_field=trg_field,
                    max_len=30,
                )
                y_train_all.append([str(ex)]) 
                predict_train_all.append([translation]) 
            print_predictions(y_train_all, predict_train_all, "train_attention2/" + varname + "/predictions/train/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_train.csv") 
            
            y_val_all = []
            predict_val_all = []
            pd_val = pd.read_csv("tokenized_data/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv", sep = ">")
            for i, ex in enumerate(list(pd_val["x"])):
                translation, _, _, _ = translate_sentence(
                    sentence=str(ex),
                    seq2seq_model=model.seq2seq,
                    src_field=src_field,
                    bos=src_field.init_token,
                    eos=src_field.eos_token,
                    eos_idx=src_field.vocab.stoi[src_field.eos_token],
                    trg_field=trg_field,
                    max_len=30,
                )
                y_val_all.append([str(ex)]) 
                predict_val_all.append([translation]) 
            print_predictions(y_val_all, predict_val_all, "train_attention2/" + varname + "/predictions/val/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_val.csv")     
    
            y_test_all = []
            predict_test_all = []
            pd_test = pd.read_csv("tokenized_data/" + varname + "/" + varname + "_test_" + str(ws_use) + ".csv", sep = ">")
            for i, ex in enumerate(list(pd_test["x"])):
                translation, _, _, _ = translate_sentence(
                    sentence=str(ex),
                    seq2seq_model=model.seq2seq,
                    src_field=src_field,
                    bos=src_field.init_token,
                    eos=src_field.eos_token,
                    eos_idx=src_field.vocab.stoi[src_field.eos_token],
                    trg_field=trg_field,
                    max_len=30,
                )
                y_test_all.append([str(ex)]) 
                predict_test_all.append([translation]) 
            print_predictions(y_test_all, predict_test_all, "train_attention2/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_test.csv")