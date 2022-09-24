import torch
from datasets import load_dataset
import numpy as np
import random
import json

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
special = ["@@NULL@@", ".", ",", ";", "!", "?", "[MASK]", "[SEP]", "[CLS]", "[PAD]"]

try:
    from .utils import * 
except Exception as ex:
    from utils import *

def hotflip(model, tokenizer, question, context, answer, answer_start, answer_end, sep_token, 
            include_answer = False, gradient_way = 'simple', number_of_flips = None):
    """
        return:
            perform hotflip and returns a dictionary
    """

    vocab = tokenizer.get_vocab()
    #vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
    invalid_indexes = []

    for k, v in vocab.items():
        if k in special or k in stop_words or k.isalnum() == False:
            invalid_indexes.append(v)
        
    
    already_generated = []
    inputs = tokenizer(question, context, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    chosen_tokens = [0] * len(tokens)
    processed_gradients, processed_saliencies, start_index, end_index = interpret(model, gradient_way, inputs, answer_start, answer_end)
    
    sep_index = tokens.index(sep_token)
    processed_saliencies[0:sep_index+1] = [0] * len(processed_saliencies[0:sep_index+1])
    
    
    answer_start = answer_start.item()
    answer_end = answer_end.item()

    if include_answer == False:
        chosen_tokens[answer_start:answer_end+1] = [2] * len(chosen_tokens[answer_start:answer_end+1])

    flips = []
    flipped_indexes = []
    steps = 0
    return_dict = None

    if sep_token == '[SEP]':
        tokenized_context = tokens[sep_index+1:len(tokens)-1]
    else:
        tokenized_context = tokens[sep_index+2:len(tokens)-1]
    indexes = list(range(0,len(tokenized_context)))
    saliencies= processed_saliencies[sep_index+1:len(tokens)-1]
    saliencies = np.array(saliencies)
    
    topk_indices = saliencies.argsort()[-10:][::-1]
    
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_context)
    topk_token_ids = []
    for i in topk_indices:
        topk_token_ids.append(token_ids[i])
    grads = processed_gradients[0][0][sep_index+1:]
    embeddings_matrix = model.bert.embeddings.word_embeddings.weight

    replacement_tokens = revised_first_order_taylor(topk_indices, topk_token_ids, invalid_indexes, grads, embeddings_matrix)
    print(tokenizer.convert_ids_to_tokens(replacement_tokens))
    

    '''
    if answer == "": 
        return_dict = {
            "question": question.lower(),
            "context" : tokenized_context,
            "answer" : answer,
            "flips" : flips,
            "total_flips": len(flips),
            "new_answer" : selected_answer,
            "indexes" : flipped_indexes
        }
        return return_dict


    while True:
        processed_saliencies, chosen_tokens, max_index = get_max(processed_saliencies, chosen_tokens)
        token_id = inputs.input_ids[0][max_index]
        embeddings_matrix = model.bert.embeddings.word_embeddings.weight
        best_random_index = first_order_taylor(max_index, processed_gradients, invalid_indexes, token_id, embeddings_matrix)
        actual_word = tokens[max_index]
        random_word = tokenizer.convert_ids_to_tokens([best_random_index])[0]
        print(tokens[46])
        if actual_word != "[SEP]":
            flips.append((actual_word, random_word))
        val = max_index-sep_index-1
        if actual_word != "[SEP]":
            flipped_indexes.append(int(val))

        tokens[max_index] = random_word
        new_input = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens[1:len(tokens)-1]), skip_special_tokens = False)
        print(new_input)
        inputs = tokenizer(new_input, return_tensors="pt")
        processed_gradients, processed_saliencies, start_index, end_index = interpret(model, gradient_way, inputs, answer_start, answer_end)
        
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        processed_saliencies[0:sep_index+1] = [0] * len(processed_saliencies[0:sep_index+1])
        if start_index < sep_index:
            selected_answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens[sep_index:end_index+1]), skip_special_tokens = True)
        else:
            selected_answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens[start_index:end_index+1]), skip_special_tokens = True)

        
        if len(selected_answer.strip()) <= 0:
           selected_answer = "Undefined Answer"
           if len(flips) == 0:
               selected_answer = answer

        if selected_answer != answer:
            break
        
        steps = steps + 1
        if number_of_flips != None and steps == number_of_flips:
            break

    inputs = tokenizer(context, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])  
    return_dict = {
            "question": question.lower(),
            "context" : tokens[1:len(tokens)-1],
            "answer" : answer,
            "flips" : flips,
            "total_flips": len(flips),
            "new_answer" : selected_answer,
            "indexes" : flipped_indexes
        }


    return return_dict
    '''
    
    

def do_hotflip(args):
    """
        return:
            perform hotflip and returns a dictionary
    """
    model, tokenizer, sep_token = get_model_tokenizer(args)
    inputs = tokenizer(args["question"], args["context"], return_tensors="pt")
    answer, start, end = get_answer(model, tokenizer, inputs)
    print(answer)
    if args["include_answer"] == 'true':
        args["include_answer"] = True
    else:
        args["include_answer"] = False
    if args['number_of_flips'] == 0:
        args['number_of_flips'] = None

    hotflip_response = hotflip(model, tokenizer, args["question"], args["context"], answer, start, end, sep_token, 
                include_answer = args["include_answer"], gradient_way = args["gradient_way"], number_of_flips = args['number_of_flips'])
    return hotflip_response


if __name__ == '__main__':

    squad = load_dataset("squad", split = "validation")
    drop = load_dataset('drop', split='validation')
    qref = load_dataset('quoref', split='validation')
    args = {
        "model_name" : "bert-base-uncased",
        "adapter" : "AdapterHub/bert-base-uncased-pf-squad_v2",
        "question" : squad[0]["question"],
        "context" : squad[0]["context"],
        "gradient_way" : "simple",
        "include_answer" : "true",
        "number_of_flips" : 10,
    }

    try:
        r = do_hotflip(args)
        print(r)
        print(r['context'][r['indexes'][0]])
        print(r['context'][r['indexes'][1]])

    except Exception as ex:
        print(ex)
    


    


    
    


