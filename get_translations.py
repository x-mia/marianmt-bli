# Importing libraries

import argparse
import torch
import io
import pandas as pd
import numpy as np
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm
from itertools import repeat

# Loading MarianMT

def load_model(src_lng, tgt_lng):
    model_name = f"Helsinki-NLP/opus-mt-{src_lng}-{tgt_lng}"

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    return model, tokenizer



# Function to get scores and translations

def get_translation(src_word, model, tokenizer):
    tgt_words = []
    scores = []
    batch = tokenizer([src_word], return_tensors="pt")
    generated_ids = model.generate(**batch, num_beams=20, do_sample=False,
                               num_return_sequences=10, max_new_tokens=10,
                               output_scores=True,return_dict_in_generate=True)
    for tok, score in zip(generated_ids.sequences, generated_ids.sequences_scores):
        tgt_word = tokenizer.decode(tok, skip_special_tokens=True)
        tgt_words.append(tgt_word)
        score = format(torch.exp(score*len(tok[tok!=62508])).numpy()*100, '.3f')
        scores.append(score)
    
    return tgt_words, scores



def translate(src_lng, eval_df, model, tokenizer):
    src_words = []
    tgt_words = []
    scors = []

    for i,row in tqdm(eval_df.iterrows(), total=eval_df.shape[0],position=0, leave=True):
        src_word = row[src_lng]
        tgt_word, score = get_translation(src_word, model, tokenizer)
        for w in tgt_word:
            tgt_words.append(w)
        for s in score:
            scors.append(s)
        src_words.extend(repeat(src_word,len(tgt_word)))
        
    return src_words, tgt_words, scors 



def making_df(src_lng, tgt_lng, src_words, tgt_words, scors):
    df = {}
    df[src_lng] = src_words
    df[tgt_lng] = tgt_words
    df["score"] = scors
    result = pd.DataFrame(df)
    result = result.drop_duplicates()
    result = result.sort_values([src_lng, 'score'])
    result = result.reset_index(drop=True)
    return result


def computing_results(result, eval_df, src_lng, tgt_lng):
    merged_df = pd.merge(result, eval_df, how='left',indicator=True, on=[src_lng, tgt_lng])
    correct = merged_df[merged_df["_merge"] == 'both']
    precision = len(correct)/len(result)
    recall = len(correct)/len(eval_df)
    f1_score = (2 * precision * recall) / (precision + recall)
    print("Precision is: ", precision)
    print("Recall is: ", recall)
    print("F1 score is: ", f1_score)
    return merged_df



def main(src_lng, tgt_lng, eval_df, output):
    print("Loading MarianMT model and tokenizer.")
    model, tokenizer = load_model(src_lng, tgt_lng)
    print("Loading evaluation dataframe.")
    eval_df = pd.read_csv(eval_df)
    print("Getting scores and translation equivalents.")
    src_words, tgt_words, scors = translate(src_lng, eval_df, model, tokenizer)
    print("Creating dataframe with results.")
    df = making_df(src_lng, tgt_lng, src_words, tgt_words, scors)
    print("Computing results...")
    df = computing_results(df, eval_df, src_lng, tgt_lng)
    print("Saving the dataframe.")
    df.to_csv(output, index=False)
    ("Done.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarianMT for BLI")
    parser.add_argument("--src_lng", type=str, help="Code of the source language")
    parser.add_argument("--tgt_lng", type=str, help="Code of the target language")
    parser.add_argument("--eval_df", type=str, help="Path to the evaluation dataframe")
    parser.add_argument("--output", type=str, help="Path to save the dataframe")

    args = parser.parse_args()

    main(args.src_lng, args.tgt_lng, args.eval_df, args.output)

