import spacy
import scispacy
from scispacy.linking import EntityLinker
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")

import numpy as np
import string

tui_id = {'T017': 5, 'T029': 6, 'T023': 7, 'T030': 8, 'T031': 9, 'T022': 10, 'T025': 11, 'T026': 12, 'T018': 13,
          'T021': 14, 'T024': 15, 'T116': 16, 'T195': 17, 'T123': 18, 'T122': 19, 'T103': 20, 'T120': 21,
          'T104': 22, 'T200': 23, 'T196': 24, 'T126': 25, 'T131': 26, 'T125': 27, 'T129': 28, 'T130': 29,
          'T197': 30, 'T114': 31, 'T109': 32, 'T121': 33, 'T192': 34, 'T127': 35, 'T190': 36, 'T049': 37,
          'T019': 38, 'T047': 39, 'T050': 40, 'T061': 41, 'T037': 42, 'T048': 43, 'T191': 44, 'T046': 45,
          'T184': 46, 'T020': 47, 'T060': 48, 'T065': 49, 'T058': 50, 'T059': 51, 'T063': 52, 'T062': 53}

def make_triples(triples, all_v = True):
    all_triples = []
    for k , vs in triples.items():
        for v in vs:
            if all_v:
                v = '%'.join(v)
            else:
                v = v[-1]
            all_triples.append('%'.join([k, v]))
    return all_triples


def process_triples(summ, client):
    processed_triples = {}
    
    for triple in client.annotate(summ):
        objs = []
        subj = triple['subject'].lower()
        subj_obj = triple['object'].lower()
        obj_add = subj_obj
        rel_add = triple['relation'].lower()
        
        if subj in processed_triples:
            objs = processed_triples[subj]
        else:
            processed_triples[subj] = []
        
        subj_obj_words = subj_obj.split()
        for rel, obj in objs:
            obj_words = obj.split()
            overlap = list(set(obj_words).intersection(subj_obj_words))
            
            obj_based = len(overlap)/len(obj_words)
            subj_obj_based = len(overlap)/len(subj_obj_words)
            
            if obj_based >= 0.5 or subj_obj_based >= 0.5:
                if subj_obj_based > obj_based:
                    objs.remove((rel, obj))
                    obj_add = subj_obj
                    rel_add = rel
                else:
                    obj_add = None
        if obj_add:
            objs.append((rel_add, obj_add))
            
        processed_triples[subj] = objs
    return processed_triples    



class AutomaticFactEval():
    
    def __init__(self):
        return
    
    def _get_umls_concepts(self, inp, all_concepts = False):
        doc = nlp(inp)
        umls_outs = {'term' : [], 'cuis' : []}
        # print("Entities: ", doc.ents)
        for entity in doc.ents:
            if len(entity._.kb_ents) == 0: continue
            # print(entity, entity._.kb_ents[0][0])

            ###################### added this
            cui = entity._.kb_ents[0][0]
            
            tui = linker.kb.cui_to_entity[cui][3]
            if tui[0] not in tui_id:
                continue
            if entity.text.lower() not in umls_outs['term']:
                umls_outs['term'].append(entity.text.lower())
            if len(entity._.kb_ents) > 0:
                umls_outs['cuis']= list(set(umls_outs['cuis'] + [entity._.kb_ents[0][0]]))
        return umls_outs
    
    def process(self, concepts):
        concepts = [each.lower().strip(string.punctuation).strip() for each in concepts]
        concepts = list(set(concepts))
        return concepts
    
    def compare(self, ref_concepts, gen_concepts):
        precision = 0
        recall = 0
        fscore = 0
        
        ## precision is out of all predicted, how many were accurate or found in ref
        true_positives = list(set(ref_concepts).intersection(set(gen_concepts)))
        if gen_concepts:
            precision = len(true_positives)/len(gen_concepts)
            
            
        ## recall is out of all in reference, how many was predicted 
        if ref_concepts:   
            recall = len(true_positives)/len(ref_concepts)
            
        if precision + recall:
            fscore =  (2 * precision * recall) / (precision + recall)
        return precision, recall, fscore
        
    def run_source_concept_faithfulness(self, ref_sums, gen_sums, use_aggregator=True):
        # df_errors = {'Evidence_Utterances': [], 'Summaries' : [], 'Generated_Summaries' : [], 'Ref_concepts' : [], 'Gen_concepts' : [], 'UMLS_score' : [],}
        all_precision_term = []
        all_recall_term = []
        all_fscore_term = []
        all_precision_cuis = []
        all_recall_cuis = []
        all_fscore_cuis = []
        
        all_gen_concepts_term = []
        all_gen_concepts_cuis = []
        
        for ref, gen in zip(ref_sums, gen_sums):         
            ref_concepts = self._get_umls_concepts(ref, all_concepts = True)
            gen_concepts = self._get_umls_concepts(gen, all_concepts = True)
            
            # print("reference concepts : ", ref_concepts)
            # print("gen concepts : ", gen_concepts)

            ref_concepts_term = ref_concepts['term']
            gen_concepts_term = gen_concepts['term']
            # ref_concepts_term = self.process(ref_concepts['term'])
            # gen_concepts_term = self.process(gen_concepts['term'])
            
            precision_term, recall_term , fscore_term = self.compare(ref_concepts_term, gen_concepts_term)
            all_precision_term += [precision_term]
            all_recall_term += [recall_term]
            all_fscore_term += [fscore_term]
            all_gen_concepts_term += [gen_concepts_term]
            
            ref_concepts_cuis = ref_concepts['cuis']
            gen_concepts_cuis = gen_concepts['cuis']
            # ref_concepts_cuis = self.process(ref_concepts['cuis'])
            # gen_concepts_cuis = self.process(gen_concepts['cuis'])
            precision_cuis, recall_cuis , fscore_cuis = self.compare(ref_concepts_cuis, gen_concepts_cuis)
            all_precision_cuis += [precision_cuis]
            all_recall_cuis += [recall_cuis]
            all_fscore_cuis += [fscore_cuis]
            all_gen_concepts_cuis += [gen_concepts_cuis]
        
        if use_aggregator:
            return {'UMLS_term_f': np.mean(all_fscore_term),
                    'UMLS_cuis_f': np.mean(all_fscore_cuis)}
                    # 'pred_concepts_term': all_gen_concepts_term,
                    # 'pred_concepts_cuis': all_gen_concepts_cuis}           

        else:
            return {'UMLS_term_f': all_fscore_term,
                    'UMLS_cuis_f': all_fscore_cuis}
                    # 'pred_concepts_term': all_gen_concepts_term,
                    # 'pred_concepts_cuis': all_gen_concepts_cuis} 