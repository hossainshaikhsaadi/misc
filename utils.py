from transformers import BertTokenizer, BertForQuestionAnswering
import torch
#from datasets import load_dataset
from torch.nn.functional import normalize
import numpy as np
import random
import json
from transformers import (
    AutoTokenizer,
    BertAdapterModel
)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def get_gradients(model, inputs, start, end):
    """Function to calculate the gradients of the input tokens

    args:
        inputs : encoded input using hugginface tokenizer
        start : answer start token index in the input
        end : answer end token index in the input
    return:
        embedding_gradient : gradient of each token
        start_ = predicted starting token index
        end_ = predicted ending token index
    """
    embedding_gradients = []
    answer_start = torch.tensor([start])
    answer_end = torch.tensor([end])
    def grad_hook(module, grad_in, grad_out):
        embedding_gradients.append(grad_out[0])
    embedding = model.bert.embeddings.word_embeddings
    handle = embedding.register_full_backward_hook(grad_hook)
    outputs = model(**inputs, start_positions=answer_start, end_positions=answer_end)
    start_ = torch.argmax(outputs.start_logits).item()
    end_ = torch.argmax(outputs.end_logits).item()
    loss = outputs.loss
    loss.backward()
    handle.remove()
    return embedding_gradients,start_,end_ 

def simple_gradient(model, inputs, start, end):
    """Function to get the saliency, start index and end index of the predicted answer using simple gradient

    args:
        model : model with adapter
        inputs : encoded input using hugginface tokenizer
        start : answer start token index in the input
        end : answer end token index in the input
    return:
        saliency : saliency score of each token
        start_ = predicted starting token index
        end_ = predicted ending token index
    """
    forward_embeddings = []
    def forward_hook(module, inputs, outputs):
        forward_embeddings.append(outputs)
    embedding = model.bert.embeddings.word_embeddings
    handle = embedding.register_forward_hook(forward_hook)
    gradients, start_, end_ = get_gradients(model, inputs, start, end)
    handle.remove()
    saliency = dot_and_normalize(gradients[0], forward_embeddings[0])
    return gradients, saliency, start_, end_

def integrated_gradient(model, inputs, start, end):
    """Function to get the saliency, start index and end index of the predicted answer using integreted gradient

    args:
        model : model with adapter
        inputs : encoded input using hugginface tokenizer
        start : answer start token index in the input
        end : answer end token index in the input
    return:
        saliency : saliency score of each token
        start_ = predicted starting token index
        end_ = predicted ending token index
    """
    grads = []
    for alpha in np.linspace(0, 1, num=10):
        def hook(module, inputs, outputs):
            outputs.mul_(torch.tensor(alpha))
        embedding = model.bert.embeddings.word_embeddings
        if alpha == 0.0 :
            forward_embeddings = embedding(inputs.input_ids)
        handle = embedding.register_forward_hook(hook)
        gradients, start_, end_ = get_gradients(model, inputs, start, end)
        grads.append(gradients)        
        handle.remove()
    
    for i in range(len(grads)):
        if i == 0:
            sum_tensor = grads[i][0]
        else:
            sum_tensor = torch.add(sum_tensor, grads[i][0])
    gradients = sum_tensor / float(len(grads))
    saliency = dot_and_normalize(gradients, forward_embeddings)
    return gradients, saliency, start_, end_

def smooth_gradient(model, inputs, start, end):
    """Function to get the saliency, start index and end index of the predicted answer using smooth gradient

    args:
        model : model with adapter
        inputs : encoded input using hugginface tokenizer
        start : answer start token index in the input
        end : answer end token index in the input
    return:
        saliency : saliency score of each token
        start_ = predicted starting token index
        end_ = predicted ending token index
    """
    grads = []
    stdev = 0.01
    for i in range(10):
        def hook(module, inputs, outputs):
            noise = torch.randn(outputs.shape, device=outputs.device) * stdev
            outputs.add_(torch.tensor(noise))
        embedding = model.bert.embeddings.word_embeddings
        if i == 0:
            forward_embeddings = embedding(inputs.input_ids)
        handle = embedding.register_forward_hook(hook)
        gradients, start_, end_ = get_gradients(model, inputs, start, end)
        grads.append(gradients)        
        handle.remove()
    for i in range(len(grads)):
        if i == 0:
            sum_tensor = grads[i][0]
        else:
            sum_tensor = torch.add(sum_tensor, grads[i][0])
    gradients = sum_tensor / float(len(grads))
    saliency = dot_and_normalize(gradients, forward_embeddings)
    return gradients, saliency, start_, end_

def interpret(model, gradient_way, inputs, start, end):
    """Function to get the saliency, start index and end index of the predicted answer using specified gradient calculation technique

    args:
        model : model with adapter
        gradient_way : gradient calculation technique (simple or integreted or smooth)
        inputs : encoded input using hugginface tokenizer
        start : answer start token index in the input
        end : answer end token index in the input
    return:
        saliency : saliency score of each token
        start_ = predicted starting token index
        end_ = predicted ending token index
    """
    if gradient_way == "simple":
        gradients, saliency, start_, end_ = simple_gradient(model, inputs, start, end)
    elif gradient_way == "integrated":
        gradients, saliency, start_, end_ = integrated_gradient(model, inputs, start, end)
    elif gradient_way == "smooth":
        gradients, saliency, start_, end_ = smooth_gradient(model, inputs, start, end)
    
    return gradients, saliency, start_, end_


def dot_and_normalize(gradients, forward_embeddings):
    """Function to calculate the saliency scores of the input tokens

    args:
        gradients : gradients of the tokens
        forward_embeddings : token embeddings extracted from the embedding layer
    
    return:
        l1_normalized_scores : for each of the tokens its gradient and embeddings is 
                               element wise multiplied then L1 norm is applied. After 
                               calculating each of the tokens saliency scores, normalization
                               is applied.
    """
    l1_normalized_scores = []
    for i in range(gradients[0].shape[0]):
        grad = gradients[0][i].view(1,gradients[0].shape[1])
        input_emb = forward_embeddings[0][i].view(1, gradients[0].shape[1])
        dot_products = (-1) * torch.mul(grad,input_emb)[0]
        l1_norm = torch.norm(dot_products, dim=0)
        l1_norm = l1_norm.item()
        #l1_norm = float("{:.2f}".format(l1_norm))
        l1_normalized_scores.append(l1_norm)
    normalization_factor = sum(l1_normalized_scores)
    #normalization_factor = float("{:.2f}".format(normalization_factor))
    for i in range(len(l1_normalized_scores)):
        l1_normalized_scores[i] = l1_normalized_scores[i]/normalization_factor
        #l1_normalized_scores[i] = float("{:.3f}".format(l1_normalized_scores[i]))
    return l1_normalized_scores
    
def get_max(saliencies, chosen_tokens):
    """Function to get the index with highest saliency

    args:
        saliencies : saliencies of each token
        choesn token : list of 0's and 1's if 0 choesn has not been chosen otherwise 1
    
    return:
        saliencies : saliencies of each token
        choesn token : list of 0's and 1's if 0 choesn has not been chosen otherwise 1
        max_index : index of the token with the highest saliency
    """
    max_index = np.argmax(saliencies)
    while chosen_tokens[max_index] == 1 or chosen_tokens[max_index] == 2:
        saliencies[max_index] = 0
        max_index = np.argmax(saliencies)
    chosen_tokens[max_index] = 1
    return saliencies, chosen_tokens, max_index

def get_min(saliencies):
    """Function to get the index with lowest saliency

    args:
        saliencies : saliencies of each token
    return:
        min_index : index of the token with the lowest saliency
    """
    #to get the index with maximum saliency score in context
    min_index = np.argmin(saliencies)
    return min_index

def get_random_token(tokens, already_generated, tokenizer, invalid_indexes):
    """Function to generate a random token

    args:
        tokens : list of all the tokens
        already_generated : list of already generated tokens
        tokenizer : tokenizer
    return:
        number : id of the token in the vocabulary
        token : token with the random id in the vocabulary
        already_generated : list of already generated tokens
    """
    number = random.randint(1, 30500)
    while number in invalid_indexes:
        number = random.randint(1, 30500)
        #token = tokenizer.convert_ids_to_tokens(number)
    #already_generated.append(number)
    return number #, token, already_generated

def saliency(model, question, context, answer_start, answer_end, gradient_way = 'simple'):
    """
        return:
            saliencies : list saliency scores of all the tokens 
    """
    inputs = tokenizer(question, context, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    saliencies, _, _ = interpret(model, gradient_way, inputs, answer_start, answer_end)
    return saliencies

def get_answer(model, tokenizer, inputs, sep_index = None):
    """
        return:
            answer : answer in plain text
            start : start token index of the answer
            end : end token index of the answer
    """
    outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits).item()
    end = torch.argmax(outputs.end_logits).item()
    if sep_index != None and int(start) < int(sep_index):
        start = sep_index
    answer_token_ids = inputs.input_ids[0][start:end+1]
    answer = tokenizer.convert_ids_to_tokens(answer_token_ids)
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer), skip_special_tokens = True)
    return answer, torch.tensor([start]), torch.tensor([end])

def get_model_tokenizer(args):
    """
        return:
            model : model with the adapter
            tokenizer : tokennizer
            sep_token : seperator token based on the model
    """
    if args["model_name"] == 'bert-base-uncased':
        model = BertAdapterModel.from_pretrained(args["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    adapter_name = model.load_adapter(args["adapter"], source="hf")
    model.active_adapters = adapter_name
    model.eval()
    if args["model_name"] == 'bert-base-uncased':
        sep_token = '[SEP]'
    elif "roberta" in args["model_name"]:
        sep_token = '</s>'
    return model, tokenizer, sep_token

def write_to_file(json_dict, file_name):
    file_to_write = open(file_name, 'w')
    json.dump(json_dict, file_to_write, indent = 4)
    file_to_write.close()

def process(tokens, saliencies, gradients, tokenizer):
    """
        return:
            processed_tokens : list of tokens with full words
            processed_saliencies : list of tokens based on processed_tokens
    """
    temp_data = [0] * len(tokens)
    current = 1
    for i in range(1,len(tokens)-1):
        if '##' in tokens[i]:
            temp_data[i-1] = current
            temp_data[i] = current
            if "##" not in tokens[i+1]: 
                current = current + 1
    current = 1 
    processed_tokens = []
    processed_saliencies = []
    processed_gradients = []
    temp_tokens = []
    temp_saliencies = []
    temp_gradients = []
    for i in range(len(tokens)):
        if temp_data[i] == 0:
            processed_tokens.append(tokens[i])
            processed_saliencies.append(saliencies[i])
            processed_gradients.append(gradients[0][0][i,:].tolist())
        if temp_data[i] == current:
            temp_tokens.append(tokens[i])
            temp_saliencies.append(saliencies[i])
            temp_gradients.append(gradients[0][0][i,:])
            if temp_data[i+1] != current:
                current = current + 1
                token_now = tokenizer.decode(tokenizer.convert_tokens_to_ids(temp_tokens), skip_special_tokens=False)
                processed_tokens.append(token_now)
                processed_saliencies.append(float(sum(temp_saliencies)/len(temp_saliencies)))
                temp_t = torch.zeros(768)
                for j in range(len(temp_gradients)):
                    temp_t = torch.add(temp_t, temp_gradients[j])
                temp_t = torch.div(temp_t, len(temp_gradients))
                processed_gradients.append(temp_t.tolist())
                temp_tokens = []
                temp_saliencies = []
                temp_gradients = []
    processed_gradients = torch.Tensor(processed_gradients)
    return processed_tokens, processed_saliencies, processed_gradients

def process_answer(tokens, tokenizer):
    """
        return:
            processed_tokens : list of tokens with full words
    """
    temp_data = [0] * len(tokens)
    current = 1
    for i in range(1,len(tokens)-1):
        if '##' in tokens[i]:
            temp_data[i-1] = current
            temp_data[i] = current
            if "##" not in tokens[i+1]: 
                current = current + 1
    current = 1 
    processed_tokens = []
    temp_tokens = []
    temp_saliencies = []

    for i in range(len(tokens)):
        if temp_data[i] == 0:
            processed_tokens.append(tokens[i])
        if temp_data[i] == current:
            temp_tokens.append(tokens[i])
            if temp_data[i+1] != current:
                current = current + 1
                token_now = tokenizer.decode(tokenizer.convert_tokens_to_ids(temp_tokens), skip_special_tokens=False)
                processed_tokens.append(token_now)
                temp_tokens = []
    return processed_tokens

def find_answer_index(processed_tokens, answer_tokens, length):
    """
        return:
            start and end of token indexes which are part of the answer
    """
    for i in range(len(processed_tokens)-length):
        j = i + length
        if processed_tokens[i:j] == answer_tokens:
            return i, j-1

def first_order_taylor(max_index, grad, invalid_indexes, token_id, embeddings_matrix):
    grad_to_emb_matrix = torch.einsum("in,kj->ik", (grad[0][0][max_index].unsqueeze(0), embeddings_matrix))
    grad_to_word_emb = torch.einsum("in,kj->ik", (grad[0][0][max_index].unsqueeze(0), embeddings_matrix[token_id, :].unsqueeze(0)))
    sub = (-1) * (grad_to_emb_matrix - grad_to_word_emb)
    sub = sub.detach().numpy()
    sub[:, invalid_indexes] = -np.inf
    best_index = sub.argmax(1)
    best_index = best_index.data[0]
    return best_index

def revised_first_order_taylor(topk_indices, topk_token_ids, invalid_indexes, grads, embeddings_matrix):
    """Function to generate replacemment tokens to flip a word

    args:

        topk_indices : indexes of the topk tokens in the context
        topk_token_ids : ids of the topk token in the vocabulary
        invalid_indexes : contains incdexes of the special tokens, stop words and subwords
        grads : gradients of the tokens of the context
        embeddings_matrix : embeddings matrix f the language model 

    returns:

        replacement_tokens : replacement tokens of the topk tokens 

    """

    already_generated = []
    replacement_tokens = []

    for i in range(len(topk_indices)):

        index = topk_indices[i]
        token_id = topk_token_ids[i]

        # multiply the gradient of the token of domension (1, 768) and the emebdding matrix of dimension (length of vocab , 768)
        grad_to_emb_matrix = torch.einsum("in,kj->ik", (grads[index].unsqueeze(0), embeddings_matrix))
        # multiply the gradient of the token of domension (1, 768) and the word emebdding of the token of dimension (1 , 768)
        grad_to_word_emb = torch.einsum("in,kj->ik", (grads[index].unsqueeze(0), embeddings_matrix[token_id, :].unsqueeze(0)))
        # subtract
        sub = (-1) * (grad_to_emb_matrix - grad_to_word_emb)
        sub = sub.detach().numpy()
        # replace invalid indexes with -inf
        sub[:, invalid_indexes] = -np.inf
        best_index = sub.argmax(1)
        best_index = best_index.data[0]

        # when the best index is already in generated indexes we need to get the second best and so on
        while best_index in already_generated:
            # replace index with -inf to get the nex best
            sub[:, best_index] = -np.inf
            best_index = sub.argmax(1)
            best_index = best_index.data[0]
        # include the new generated token index in the already generated list
        already_generated.append(best_index)
        # append it in the replacement tokens
        replacement_tokens.append(best_index)

    return replacement_tokens
