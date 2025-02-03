import os
import re
import json
import faiss
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import torch 
from sentence_transformers import SentenceTransformer, util
import importlib.util
import os
from datetime import datetime

from timeit import default_timer as timer

import langid
import langcodes

import requests
from bs4 import BeautifulSoup

from langdetect import detect

from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline
)
from transformers.pipelines import AggregationStrategy
import numpy as np

import spacy
import faiss

nltk.download('wordnet')
nltk.download('omw-1.4')

class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])

class FactMap():
    def __init__(self, df_facts, df_posts):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = list(stopwords.words('english'))+['list', 'extracted', 'key', 'entities']
        self.facts = df_facts
        self.posts = df_posts
        self.processor = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentence_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True, device=self.processor)
        self.extractor = KeyphraseExtractionPipeline(model="ml6team/keyphrase-extraction-kbir-kpcrowd")
        self.NER = spacy.load("en_core_web_sm")

    def clean_facts(self, fact_id_index):
        possible_facts = {}
        language_indexes = {}
        language_facts = ['eng', 'fra', 'deu', 'por', 'spa', 'tha', 'msa', 'ara']
        faiss_facts = {}
        res = faiss.StandardGpuResources()

        for index, fact in enumerate(self.facts["claim"]):

            try:
                df_fact = self.facts.iloc[index]
                joined_fact = f"{fact[1]}. {df_fact["title"][1]}"
            except:
                joined_fact = fact[1]

            clean_fact = re.sub(r'[^\w\s]', '', joined_fact)

            fact_id = fact_id_index[index]

            # try:
            #     fact_language = detect(fact[0])
            # except:
            #     fact_language = "Unknown"
        
            # # fact_language, confidence = langid.classify(fact[0])

            # language_name = langcodes.Language.make(language=fact_language).display_name()

            language_name = fact[2][0][0]

            if len(language_name) == 1:
                language_name = fact[2][0]

            if language_name  not in possible_facts:
                possible_facts[language_name] = []
            if language_name  not in language_indexes:
                print(language_name)
                language_indexes[language_name] = []

            possible_facts[language_name].append(clean_fact)
            language_indexes[language_name].append(fact_id)

        for language in possible_facts.keys():
            print("Language:", language)
            print(len(possible_facts[language]))

            encoded_fact = self.sentence_model.encode(possible_facts[language], device=self.processor, convert_to_tensor=False, show_progress_bar=True)

            dimension = encoded_fact[0].shape[0]
            print("Dimension:", dimension)
            index_cpu = faiss.IndexFlatL2(dimension)

            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

            index.add(np.array(encoded_fact))
                          
            faiss_facts[language] = index

        all_index = faiss.IndexFlatL2(dimension)

        for language in possible_facts.keys():
            all_index.add(faiss_facts[language].reconstruct_n(0, faiss_facts[language].ntotal))

        return faiss_facts, language_facts, language_indexes, all_index, dimension
    
    def similar_fact(self, fa_embedding, post_text):
        facts = []
        sim_dist = []

        post_embedding = self.sentence_model.encode(post_text, device=self.processor)
        post_embedding = post_embedding.reshape(1, -1)
        
        k = 250
        distances, indicies = fa_embedding.search(post_embedding, k)

        top100_facts = [(idx, distances[0][i]) for i, idx in enumerate(indicies[0])]

        for fact, sim in top100_facts:
            facts.append(fact)
            sim_dist.append(sim)

        return facts, sim_dist
    
    def align_facts(self, post_date, similar_facts, fact_indexes):

        filtered_facts = []

        print("Amount of Similar Facts:", len(similar_facts))
        print(similar_facts[:10])

        if post_date != None:

            for num in similar_facts:

                index_fact_id = self.facts.index.to_list().index(int(fact_indexes[num]))

                timestamp = df_facts["instances"].iloc[index_fact_id][0][0]

                if timestamp != None:

                    if isinstance(timestamp, str):
                        dt_object = datetime.fromisoformat(timestamp)
                        timestamp = int(dt_object.timestamp())

                    if abs(timestamp - post_date) < (9*30*24*60*60):

                        filtered_facts.append(num)

                else:
                    if similar_facts.index(num)/len(similar_facts) <= 0.30:
                        print(similar_facts.index(num)/len(similar_facts))
                        filtered_facts.append(num)
        else: 

            filtered_facts = similar_facts[:10]

        print("Amount of Filtered Facts:", len(filtered_facts))

        if len(filtered_facts) < 10:
            for value in similar_facts:
                if value not in filtered_facts:
                    filtered_facts.append(value)
                if len(filtered_facts) == 10:  
                    break

        print(filtered_facts[:10])
            
        return filtered_facts[:10]
    

def load_data():
    directory = os.path.dirname(__file__)
    load_data_path = os.path.join(directory, "Test_Data", "load.py")

    spec = importlib.util.spec_from_file_location("load", load_data_path)
    load = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(load)

    return load.df_fact_checks, load.df_posts, load.df_fact_check_post_mapping


if __name__ == "__main__":

    start = timer()

    file = "Test_Data/crosslingual_predictions.json"

    with open(file, "r") as json_file:
        predictions = json.load(json_file)

    post_id_list = list(predictions.keys())

    df_facts, df_posts, df_mapping = load_data()

    fact_mapper = FactMap(df_facts, df_posts)

    # evaluate = Evaluate(df_facts, df_posts, df_mapping, 0)
    fact_id_index = df_facts.index.to_list()

    facts_embedding, languages, language_index, total_embedding, fiass_dimension = fact_mapper.clean_facts(fact_id_index)

    total_index = []

    for lang in language_index.keys():
        total_index.extend(language_index[lang])

    print("Length of Total Index:",len(total_index))

    for i, post_id in enumerate(post_id_list):
        final_index = []
        fact_indexes = []

        # To use the index in df_posts of the post_id
        index_post_id = df_posts.index.to_list().index(int(post_id))

        post = df_posts["text"].iloc[index_post_id]
        ocr = df_posts["ocr"].iloc[index_post_id]
        post_dt = df_posts["instances"].iloc[index_post_id][0][0]

        if isinstance(post_dt, str):  
            dt_object = datetime.fromisoformat(post_dt)
            post_dt= int(dt_object.timestamp())

        post_lang = []

        # Finding the valid textual information for each post, either in the text, ocr, or in both sections of the row.
        print(post_id)
        if post or ocr:
            if post and ocr:
                org_text = f"{post[0]}. {ocr[0][0]}"
                post_text = f"{post[1]}. {ocr[0][1]}"
                post_lang = post[2] + ocr[0][2]
            elif post:
                org_text = post[0]
                post_text = post[1]
                post_lang = post[2]
            elif ocr:
                org_text = ocr[0][0]
                post_text = ocr[0][1]
                post_lang = ocr[0][2]

            language_name = post_lang[0][0]

            if len(language_name) == 1:
                language_name = post_lang[0]

            if language_name in language_index.keys():

                print(language_name)

                sim_fact_ids, sim_val = fact_mapper.similar_fact(facts_embedding[language_name], post_text)

                print("Similarity Values:", sim_val[:10])

                fact_indexes = language_index[language_name]

                if len(sim_fact_ids) < 10:
                    print("UNDER 10 FACTS")
                    sim_fact_ids, sim_val = fact_mapper.similar_fact(total_embedding, post_text)

                    fact_indexes = total_index
            else:

                print("Language Not:", language_name)

                sim_fact_ids, sim_val = fact_mapper.similar_fact(total_embedding, post_text)

                fact_indexes = total_index

                print("Similarity Values:", sim_val[:10])


            print("Similar Facts:", len(sim_fact_ids))

            if post_dt:
                facts = fact_mapper.align_facts(post_dt, sim_fact_ids, fact_indexes)
            else:
                facts = sim_fact_ids[:10]

            print("Filtered Facts:", len(facts))

            for num in facts:

                final_index.append(fact_indexes[num])

            predictions[post_id] = final_index

            print(predictions[f"{post_id}"])

            print(f"{round(i/len(post_id_list) * 100, 2)}%")
            print()

end = timer()
print(end-start)

with open("crosslingual_predictions.json", "w") as json_file:
    json.dump(predictions, json_file)
