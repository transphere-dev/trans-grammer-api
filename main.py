import json
import re
import time

import errant
import pandas as pd
import torch
from annotated_text import annotated_text
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gramformer import Gramformer
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import datetime

annotator = errant.load('en')

print("Loading models...")

app = FastAPI()


class Sentences(BaseModel):
    sentences: list


origins = [

    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"
correction_model_tag = "prithivida/grammar_error_correcter_v1"
correction_tokenizer = AutoTokenizer.from_pretrained(correction_model_tag)
correction_model = AutoModelForSeq2SeqLM.from_pretrained(correction_model_tag)

influent_sentences = [
    "I is dog."
]


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


print("Models loaded !")

gf = Gramformer(models=1, use_gpu=False)  # 1=corrector, 2=detector


@app.get("/")
def read_root():
    return {"Gramformer !"}


# @app.get("/{correct}")
# def get_correction(input_sentence):
#     set_seed(1212)
#     scored_corrected_sentence = correct(input_sentence)
#     return {"scored_corrected_sentence": scored_corrected_sentence}

@app.post("/sentence")
def get_corrected_sentence(sentences: Sentences):
    sentence_list = sentences.sentences
    # print(sentence_list)
    set_seed(1212)

    scored_corrected_sentence = correct_sentence(sentence_list)
    sent_list = list(scored_corrected_sentence)

    highlighted_sentences = show_highlights(sentence_list[0], sent_list[0])
    # print(highlighted_sentences)

    return json.dumps({'corrected_sentence': highlighted_sentences})


def correct_sentence(input_sentence):
    for influent_sentence in input_sentence:
        """
        Correct influent_sentences
        
        """
        corrected_sentences = gf.correct(influent_sentence, max_candidates=3)
        # print("[Input] ", influent_sentence)
        # for corrected_sentence in corrected_sentences:
        #   print("[Correction] ",corrected_sentence)
        #   print("[Edits] ", gf.highlight(influent_sentence, corrected_sentence))
        return corrected_sentences


def correct(input_sentence, max_candidates=1):
    correction_prefix = "gec: "
    input_sentence = correction_prefix + input_sentence
    input_ids = correction_tokenizer.encode(input_sentence, return_tensors='pt')
    input_ids = input_ids.to(device)

    preds = correction_model.generate(
        input_ids,
        do_sample=True,
        max_length=128,
        top_k=50,
        top_p=0.95,
        #        num_beams=7,
        early_stopping=True,
        num_return_sequences=max_candidates)

    corrected = set()
    for pred in preds:
        corrected.add(correction_tokenizer.decode(pred, skip_special_tokens=True).strip())

    corrected = list(corrected)
    return (corrected[0], 0)  # Corrected Sentence, Dummy score


def show_highlights(input_text, corrected_sentence):
    try:
        strikeout = lambda x: '\u0336'.join(x) + '\u0336'
        highlight_text = highlight(input_text, corrected_sentence)
        print(highlight_text)
        color_map = {'d': '#faa', 'a': '#afa', 'span': '#fea'}
        tokens = re.split(r'(<[das]\s.*?<\/[das]>)', highlight_text)
        # print(tokens)
        annotations = []  # ['Sorry i ', ('forgot', 'VERB:TENSE', '#fea'), ' how to write. ', ('Tomorrow', 'SPELL', '#fea'), ' i ', ('remember.', 'VERB', '#fea'), '']
        for token in tokens:
            soup = BeautifulSoup(token, 'html.parser')
            tags = soup.findAll()

            if tags:
                _tag = tags[0].name
                _type = tags[0]['type']
                _text = tags[0]['edit']
                _color = color_map[_tag]

                if _tag == 'd':
                    _text = strikeout(tags[0].text)

                annotations.append((_text, _type, _color))
            else:
                annotations.append(token)
        annotated_text(*annotations)

        return highlight_text

    except Exception as e:
        print('Some error occured!' + str(e))


def show_edits(input_text, corrected_sentence):
    try:
        edits = get_edits(input_text, corrected_sentence)
        df = pd.DataFrame(edits, columns=['type', 'original word', 'original start', 'original end', 'correct word',
                                          'correct start', 'correct end'])
        df = df.set_index('type')

    except Exception as e:
        print('Some error occured!' + str(e))


def highlight(orig, cor):
    edits = _get_edits(orig, cor)
    orig_tokens = orig.split()

    ignore_indexes = []

    for edit in edits:
        edit_type = edit[0]
        edit_str_start = edit[1]
        edit_spos = edit[2]
        edit_epos = edit[3]
        edit_str_end = edit[4]

        # if no_of_tokens(edit_str_start) > 1 ==> excluding the first token, mark all other tokens for deletion
        for i in range(edit_spos + 1, edit_epos):
            ignore_indexes.append(i)

        if edit_str_start == "":
            if edit_spos - 1 >= 0:
                new_edit_str = orig_tokens[edit_spos - 1]
                edit_spos -= 1

            else:
                new_edit_str = orig_tokens[edit_spos + 1]
                edit_spos += 1

            if edit_type == "PUNCT":
                timestamp = str(datetime.datetime.timestamp(datetime.datetime.now())).replace('.', '-')

                st = "<a id=" + timestamp + " " + "type='" + edit_type + "' edit='" + \
                     edit_str_end + "'>" + new_edit_str + "</a>"
            else:
                timestamp = str(datetime.datetime.timestamp(datetime.datetime.now())).replace('.', '-')

                st = "<a id=" + timestamp + " " + "type='" + edit_type + "' edit='" + new_edit_str + \
                     " " + edit_str_end + "'>" + new_edit_str + "</a>"
            orig_tokens[edit_spos] = st
        elif edit_str_end == "":
            timestamp = str(datetime.datetime.timestamp(datetime.datetime.now())).replace('.', '-')

            st = "<d id=" + timestamp + " " + "type='" + edit_type + "' edit=''>" + edit_str_start + "</d>"
            orig_tokens[edit_spos] = st
        else:
            timestamp = str(datetime.datetime.timestamp(datetime.datetime.now())).replace('.', '-')

            st = "<span id=" + timestamp + " " + "type='" + edit_type + "' edit='" + \
                 edit_str_end + "'>" + edit_str_start + "</span>"
            orig_tokens[edit_spos] = st

    for i in sorted(ignore_indexes, reverse=True):
        del (orig_tokens[i])
    return (" ".join(orig_tokens))


def _get_edits(orig, cor):
    orig = annotator.parse(orig)
    cor = annotator.parse(cor)
    alignment = annotator.align(orig, cor)
    edits = annotator.merge(alignment)

    if len(edits) == 0:
        return []

    edit_annotations = []
    for e in edits:
        e = annotator.classify(e)
        edit_annotations.append((e.type[2:], e.o_str, e.o_start, e.o_end, e.c_str, e.c_start, e.c_end))

    if len(edit_annotations) > 0:
        return edit_annotations
    else:
        return []


def get_edits(orig, cor):
    return _get_edits(orig, cor)

# def set_seed(seed):
#   torch.manual_seed(seed)
#   if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
#
# set_seed(1212)
#
#
# gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector
#
#
#
# influent_sentences = [
#     "He are moving here.",
#     "I am doing fine. How is you?",
#     "How is they?",
#     "Matt like fish",
#     "the collection of letters was original used by the ancient Romans",
#     "We enjoys horror movies",
#     "Anna and Mike is going skiing",
#     "I walk to the store and I bought milk",
#     " We all eat the fish and then made dessert",
#     "I will eat fish for dinner and drink milk",
#     "what be the reason for everyone leave the company",
# ]
#
# for influent_sentence in influent_sentences:
#     corrected_sentences = gf.correct(influent_sentence, max_candidates=1)
#     print("[Input] ", influent_sentence)
#     for corrected_sentence in corrected_sentences:
#       print("[Correction] ",corrected_sentence)
#     print("-" *100)
