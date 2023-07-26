import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from graphviz import Digraph
import pandas as pd
from collections import defaultdict
from queue import Queue


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Check if triple is similar to some triple in current kg
def check_similar(triple, kg):
    subject = triple['subject'].split()
    relation = triple['relation'].split()
    object = triple['object'].split()
    for fact in kg:
        similar_count = 0
        for s in subject:
            if s in fact['subject']:
                similar_count += 1
                break
        for r in relation:
            if r in fact['relation']:
                similar_count += 1
                break
        for o in object:
            if o in fact['object']:
                similar_count += 1
                break
        if similar_count >= 3:
            return fact
    return None


# Knowledge Graph Construction
def openie_coref(client, input_text):
    story_kg_dataset = {}
    # set the options
    CUSTOM_PROPS = {
        'annotators': 'openie, coref',
        'openie.resolve_coref': True,
        'openie.affinity_probability_cap': 0.2,
    }
    # text to knowledge graph
    story_kg = []
    ann = client.annotate(input_text, properties=CUSTOM_PROPS)
    sentence_idx = 0
    for sentence in ann.sentence:
        sentence_text = ' '.join([token.word for token in sentence.token])
        # pos_dict = {token.word: token.pos for token in sentence.token}
        for triple in sentence.openieTriple:
            current_triple = {
                    'subject': triple.subject.lower(),
                    'relation': triple.relation.lower(),
                    'object': triple.object.lower(),
                    'raw_text': sentence_text,
                    'sentence_idx': sentence_idx
                }
            similar_triple = check_similar(current_triple, story_kg)
            if similar_triple is None \
                and len(current_triple['subject']) > 0 \
                    and len(current_triple['relation']) > 0 \
                        and len(current_triple['object']) > 0:
                story_kg.append(current_triple)
            else:
                triple_text = ' '.join([current_triple['subject'], current_triple['relation'], current_triple['object']])
                similar_triple_text = ' '.join([similar_triple['subject'], similar_triple['relation'], similar_triple['object']])
                if len(triple_text) > len(similar_triple_text) and current_triple['sentence_idx'] == similar_triple['sentence_idx']:
                    story_kg.remove(similar_triple)
                    story_kg.append(current_triple)
                else:
                    pass
        sentence_idx += 1
    return story_kg


# class of fact
class Fact:
    def __init__(self, subject: str, relation: str, object: str, sentence_idx=None, status=None):
        self.subject = subject
        self.relation = relation
        self.object = object
        self.sentence_idx = sentence_idx
        self.status = status
    
    def __str__(self):
        return f'[{self.subject}, {self.relation}, {self.object}]'
    
    def __repr__(self):
        return f'[{self.subject}, {self.relation}, {self.object}]'


# class of knowledge graph
class KnowledgeGraph:
    def __init__(self, device, model, tokenizer, story_kg_file=None):
        self.device = device
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.tokenizer = tokenizer

        self.kg = self.load_kg(story_kg_file)
        self.kg_embedding = self.create_kg_embedding()

        self.graph = defaultdict(list)
        self.nodes = set()
        for fact in self.kg:
            self.add_fact(fact.subject, fact.relation, fact.object)
            self.add_fact(fact.object, fact.relation, fact.subject)
    
    def add_fact(self, subject, relation, obj):
        subject = subject.lower()
        relation = relation.lower()
        obj = obj.lower()
        self.graph[subject].append((relation, obj))
        self.nodes.add(subject)
        self.nodes.add(obj)

    def find_shortest_distance(self, start_node, end_node):
        start_node = start_node.lower()
        end_node = end_node.lower()

        if start_node not in self.nodes or end_node not in self.nodes:
            return -1
        
        visited = set()
        queue = Queue()
        queue.put((start_node, 0))

        while not queue.empty():
            node, distance = queue.get()
            visited.add(node)
            if node == end_node:
                return distance
            for _, neighbor in self.graph[node]:
                if neighbor not in visited:
                    queue.put((neighbor, distance + 1))
        return -1


    def find_two_sentence_distance(self, corenlp_client, sentence1, sentence2):
        sentence1_kg = openie_coref(corenlp_client, sentence1)
        sentence2_kg = openie_coref(corenlp_client, sentence2)
        distance_score = 0
        for kg1 in sentence1_kg:
            for kg2 in sentence2_kg:
                distance = self.find_shortest_distance(kg1['subject'], kg2['subject'])
                if distance != -1:
                    distance_score += 1
        return distance_score


    def load_kg(self, story_kg_file):
        if story_kg_file.endswith('.tsv'):
            kg = self.load_kg_tsv(story_kg_file)
        elif story_kg_file.endswith('.json'):
            kg = self.load_kg_json(story_kg_file)
        else:
            assert False, 'story_kg_file must be json or tsv file'
        return kg

    def load_kg_tsv(self, kg_tsv_file):
        kg_tsv = pd.read_csv(kg_tsv_file, sep='\t')
        kg = []
        for idx, row in kg_tsv.iterrows():
            kg.append(Fact(row['subject'], row['relation'], row['object'], row['sentence_idx'], None))
        return kg

    def load_kg_json(self, kg_json_file):
        with open(kg_json_file, encoding='utf8') as f:
            kg_json = json.load(f)
        # for each fact in kg_json, create a Fact object
        kg = []
        for fact in kg_json:
            kg.append(Fact(fact['subject'], fact['relation'], fact['object'], fact['sentence_idx'], None))
        return kg

    def get_kg_string_list(self) -> list:
        kg_string_list = []
        for fact in self.kg:
            kg_string_list.append(str(fact))
        return kg_string_list
    
    def set_kg_status(self, fact_subject, fact_relation, fact_object, status):
        # find fact in kg, set status and return set count
        # if input is None -> ignore
        set_count = 0
        for f in self.kg:
            if fact_subject is not None and fact_subject != f.subject:
                continue
            if fact_relation is not None and fact_relation != f.relation:
                continue
            if fact_object is not None and fact_object != f.object:
                continue
            f.status = status
            set_count += 1
        return set_count
    
    def set_kg_status_by_facts(self, facts, status):
        # find fact in kg, set status and return set count
        # if input is None -> ignore
        set_count = 0
        for f in self.kg:
            if f in facts:
                f.status = status
                set_count += 1
        return set_count

    
    def filter_kg(self, fact_subject, fact_relation, fact_object, status, fuzzy=False):
        # return a list of fact object which having the same subject, relation, object and status
        filtered_kg = []
        # lower case if not none
        if fact_subject is not None:
            fact_subject = fact_subject.lower()
        if fact_relation is not None:
            fact_relation = fact_relation.lower()
        if fact_object is not None:
            fact_object = fact_object.lower()

        for fact in self.kg:
            # fuzzy match
            if fuzzy:
                if fact_subject is not None and fact_subject not in fact.subject.lower():
                    continue
                if fact_relation is not None and fact_relation not in fact.relation.lower():
                    continue
                if fact_object is not None and fact_object not in fact.object.lower():
                    continue
            # exact match (not care about case)
            else:
                if fact_subject is not None and fact_subject != fact.subject.lower():
                    continue
                if fact_relation is not None and fact_relation != fact.relation.lower():
                    continue
                if fact_object is not None and fact_object != fact.object.lower():
                    continue
            if status is not None and fact.status != status:
                continue
            filtered_kg.append(fact)
        return filtered_kg

    def filter_kg_by_sentence_idx(self, fact_subject, fact_relation, fact_object, status, sentence_idx ,fuzzy=False, after_idx=True):
        pre_filtered_kg = self.filter_kg(fact_subject, fact_relation, fact_object, status, fuzzy)
        if sentence_idx == -1:
            return pre_filtered_kg
        filtered_kg = []
        if after_idx:
            for fact in pre_filtered_kg:
                if fact.sentence_idx >= sentence_idx:
                    filtered_kg.append(fact)
        else:
            for fact in pre_filtered_kg:
                if fact.sentence_idx < sentence_idx:
                    filtered_kg.append(fact)
        return filtered_kg

    def sentenece_embedding(self, sentences):
        encoded_input = self.tokenizer(sentences, 
                                       padding=True, 
                                       truncation=True, 
                                       return_tensors='pt').to(self.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def create_kg_embedding(self):
        # use sbert to create fact embedding
        # fact_sentences = ''.join([str(fact) for fact in self.kg]) 
        fact_list = [f'{fact.subject} {fact.relation} {fact.object}' for fact in self.kg]
        return self.sentenece_embedding(fact_list)

    def sentence_match_fact(self, sentence, k):
        input_sentence_embeddings = self.sentenece_embedding([sentence])
        # use SBERT to find top k match fact
        # find top k similar sentences
        similarities = F.cosine_similarity(input_sentence_embeddings, self.kg_embedding)
        topk_idx_list = torch.topk(similarities, k=k).indices.tolist()
        # decode the topk sentences
        topk_facts = [self.kg[i] for i in topk_idx_list]
        return topk_idx_list, topk_facts
