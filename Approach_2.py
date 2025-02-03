import os
import re
import json
import faiss
import Levenshtein
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import torch 
from sentence_transformers import SentenceTransformer, util
import importlib.util
import os
import random
import datetime

from timeit import default_timer as timer

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
        self.fiass_res = faiss.StandardGpuResources()

    def index_facts(self):

        possible_facts = {}
        language_indexes = {}
        embedded_facts = {}
        all_indexes = []

        for index, fact in enumerate(self.facts["claim"]):

            try:
                df_fact = self.facts.iloc[index]
                joined_fact = f"{fact[1]}. {df_fact["title"][1]}"
            except:
                joined_fact = fact[1]

            clean_fact = re.sub(r'[^\w\s]', '', joined_fact)

            fact_id = fact_id_index[index]

            fact_language = df_fact["claim"][2][0][0]

            if fact_language not in possible_facts:
                possible_facts[fact_language] = []
            if fact_language not in language_indexes:
                language_indexes[fact_language] = []

            possible_facts[fact_language].append(clean_fact)
            language_indexes[fact_language].append(fact_id)

            all_indexes.append(fact_id)

        for language in possible_facts.keys():

            print("Language:", language)

            encoded_fact = self.sentence_model.encode(possible_facts[language], device=self.processor, convert_to_tensor=False, show_progress_bar=True)

            dimension = encoded_fact[0].shape[0]

            index_cpu = faiss.IndexFlatL2(dimension)

            index = faiss.index_cpu_to_gpu(self.fiass_res, 0, index_cpu)

            index.add(np.array(encoded_fact))
                          
            embedded_facts[language] = index

        embedded_all = faiss.IndexFlatL2(dimension)

        for language in possible_facts.keys():
            embedded_all.add(embedded_facts[language].reconstruct_n(0, embedded_facts[language].ntotal))

        return embedded_facts, language_indexes, embedded_all, all_indexes, possible_facts.keys()
    
    def similar_fact(self, fact_embedding, embedded_fact_ids, post_text):

        post_embedding = self.sentence_model.encode(post_text, device=self.processor)
        post_embedding = post_embedding.reshape(1, -1)
        
        distances, indicies = fact_embedding.search(post_embedding, 10)

        top10_facts = [embedded_fact_ids[fact_ids] for fact_ids in indicies[0]]

        return top10_facts
    
    def filter_facts(self, fact_ids, post_date, time_window = 6*30*24*60*60):
        filtered_facts = []

        for fact_id in fact_ids:

            # fact_id -> df_facts index to retrieve all 
            df_fact = self.facts.iloc[fact_ids.index(int(fact_id))]

            fact_date = df_fact["instances"][0][0]

            if fact_date != None and post_date != None:

                if isinstance(fact_date, str):  
                    dt_object = datetime.fromisoformat(fact_date)
                    fact_date= int(dt_object.timestamp())

                if abs(post_date - fact_date) <= time_window:

                    filtered_facts.append(fact_id)
            
            else:

                filtered_facts.append(fact_id)
                
        return filtered_facts

def load_data():
    directory = os.path.dirname(__file__)
    load_data_path = os.path.join(directory, "Dev_Data", "load.py")

    spec = importlib.util.spec_from_file_location("load", load_data_path)
    load = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(load)

    return load.df_fact_checks, load.df_posts, load.df_fact_check_post_mapping


if __name__ == "__main__":

    start = timer()

    file = "Dev_Data/monolingual_predictions.json"

    with open(file, "r") as json_file:
        predictions = json.load(json_file)

    post_id_list = list(predictions.keys())

    df_facts, df_posts, df_mapping = load_data()

    fact_mapper = FactMap(df_facts, df_posts)

    fact_id_index = df_facts.index.to_list() # fact_ids

    embed_facts, embed_facts_index, embed_all, index_all, languages = fact_mapper.index_facts()

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

        # Finding the valid textual information for each post, either in the text, ocr, or in both sections of the row.

        post_lang = []

        if post or ocr:
            if post and ocr:
                post_text = f"{post[1]}. {ocr[0][1]}"
                post_lang = post[2][0][0] + ocr[0][2][0][0]
            elif post:
                post_text = post[1]
                post_lang = post[2][0][0]
            elif ocr:
                post_text = ocr[0][1]
                post_lang = ocr[0][2][0][0]

            languages_list = [item[0] for item in post_lang]
            languages_filtered = [item for item in languages_list if item in languages]

            languages_filtered = list(set(languages_filtered))

            if post_lang in embed_facts_index.keys():
                embedding_indexes = embed_facts_index[post_lang]

                embeddings = embed_facts[post_lang]

            else:
                embedding_indexes = index_all

                embeddings = embed_all

            final_index = fact_mapper.similar_fact(embeddings, embedding_indexes, post_text)

            predictions[post_id] = final_index

            print(predictions[f"{post_id}"])

            print(f"{round(i/len(post_id_list) * 100, 2)}%")

end = timer()
print(end-start)

with open("monolingual_predictions.json", "w") as json_file:
    json.dump(predictions, json_file)