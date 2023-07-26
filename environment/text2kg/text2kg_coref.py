import os
import json
from stanza.server import CoreNLPClient
from graphviz import Digraph
import pandas as pd
import logging
import argparse
from datetime import datetime


# stanza.install_corenlp()

# set logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# set the hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input story json file path", type=str, default='../../data/story_sentence_train.json')
parser.add_argument("-o", "--output", help="output file path", type=str, default='data/kg2text/kg2text_coref_train.json')
parser.add_argument("-g", "--goutput", help="output knowledge graph file path", type=str, default='data/kg/train_coref')
parser.add_argument("-f", "--coreference", help="coreference resolution", type=bool, default=True)
args = parser.parse_args()


# filter rules
def check_pos(triple, pos_dict):
    return True
    subject = triple.subject.split()
    relation = triple.relation.split()
    object = triple.object.split()

    # object is identical to subject
    if triple.subject == triple.object:
        return False
    # subject does not contain a noun
    if not any([pos_dict.get(word, '').startswith('NN') for word in subject]):
        return False
    # The first word in object is identical to the last word in relation
    if relation[-1].lower() == object[0].lower():
        return False
    # subject or object starts with a pronoun or conjunction
    if pos_dict[subject[0]] == 'PRP' or \
        pos_dict[subject[0]] == 'PP$' or \
        pos_dict[subject[0]] == 'WP' or \
        pos_dict[subject[0]] == 'WP$' or \
        pos_dict[subject[0]] == 'CC':
        return False
    # There are no verbs present in the subject/relation/object
    if not any([pos_dict.get(word, '').startswith('VB') for word in subject]) and \
        not any([pos_dict.get(word, '').startswith('VB') for word in relation]) and \
        not any([pos_dict.get(word, '').startswith('VB') for word in object]):
        return False
    return True


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


# create graphviz
def create_graphviz(triple_list, story_name):
    dot = Digraph(comment=f'{story_name}')
    for fact in triple_list:
        dot.node(fact['subject'], fact['subject'])
        dot.node(fact['object'], fact['object'])
        dot.edge(fact['subject'], fact['object'], fact['relation'])
    dot.format = 'png'
    dot.render(os.path.join(f'{args.goutput}', f'{story_name}.gv'))


# create tsv
def create_tsv(triple_list, story_name):
    df = pd.DataFrame(triple_list, columns=['subject', 'relation', 'object', 'raw_text', 'sentence_idx'])
    df.to_csv(os.path.join(f'{args.goutput}', f'{story_name}.tsv'), sep='\t', index=False)


def create_kg2text_dataset(story_kg_dataset):
    kg2text_dataset = {
        'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data': []
    }

    id = 0
    for story_name in story_kg_dataset.keys():
        current_kg2text_dataset = []
        done_idx_list = []
        for fact in story_kg_dataset[story_name]:
            if fact['sentence_idx'] not in done_idx_list:
                current_kg2text_dataset.append({
                    'id': id,
                    'story_name': story_name,
                    'sentence_idx': fact['sentence_idx'],
                    'raw_text': fact['raw_text'],
                    'triple': [
                        {
                        'subject': fact['subject'],
                        'relation': fact['relation'],
                        'object': fact['object']
                        }
                    ]
                })
                done_idx_list.append(fact['sentence_idx'])
                id += 1
            else:
                # find the same sentence_idx in story_dataset, than append the triple
                for current_kg2text_data in current_kg2text_dataset:
                    if fact['sentence_idx'] == current_kg2text_data['sentence_idx']:
                        current_kg2text_data['triple'].append({
                                'subject': fact['subject'],
                                'relation': fact['relation'],
                                'object': fact['object']
                            })
                        # sort by the subject appear order (ignore the case)
                        current_kg2text_data['triple'] = sorted(current_kg2text_data['triple'], key=lambda x: fact['raw_text'].lower().find(x['subject']))
                        break

        kg2text_dataset['data'].extend(current_kg2text_dataset)

    with open(os.path.join(f'{args.output}'), 'w', encoding='utf8') as f:
        json.dump(kg2text_dataset, f, indent=4)


# create json
def create_json(triple_list, story_name):
    with open(os.path.join(f'{args.goutput}', f'{story_name}.json'), 'w', encoding='utf8') as f:
        json.dump(triple_list, f, indent=4)


# Knowledge Graph Construction
def knowledge_graph_construction(story_dataset):
    story_kg_dataset = {}
    # set the options
    CUSTOM_PROPS = {
        'annotators': 'openie, pos, coref',
        'openie.resolve_coref': args.coreference,
        'openie.affinity_probability_cap': 0.2,
    }
    
    with CoreNLPClient(be_quiet=True) as client:
        # openie resolve_coref =
        for story_name in story_dataset.keys():
            logging.info(f'Processing {story_name}')
            # load the story
            story = story_dataset[story_name]
            story_text = ' '.join(story)
            
            # text to knowledge graph
            story_kg = []
            ann = client.annotate(story_text, properties=CUSTOM_PROPS)
            sentence_idx = 0
            for sentence in ann.sentence:
                sentence_text = ' '.join([token.word for token in sentence.token])
                pos_dict = {token.word: token.pos for token in sentence.token}
                for triple in sentence.openieTriple:
                    # check if all relation words are in the pos_dict
                    # relation_words = triple.relation.split()
                    # if not all([word in pos_dict for word in relation_words]):
                    #     relation_ann = client.annotate(triple.relation)
                    #     for token in relation_ann.sentence[0].token:
                    #         pos_dict[token.word] = token.pos
                    # filter rules
                    if check_pos(triple, pos_dict):
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

            story_kg_dataset[story_name] = story_kg

            # to graphviz
            create_graphviz(story_kg, story_name)

            # to tsv
            create_tsv(story_kg, story_name)

            # to json
            create_json(story_kg, story_name)

        create_kg2text_dataset(story_kg_dataset)


if __name__ == '__main__':
    # open the story json file
    with open(args.input, 'r', encoding='utf8') as f:
        story_dataset = json.load(f)

    logging.info('Start to construct knowledge graph')
    knowledge_graph_construction(story_dataset)
    logging.info('Finish constructing knowledge graph')
