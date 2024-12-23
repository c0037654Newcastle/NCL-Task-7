# Possible Approach Based on FACT-GPT: Fact-Checking Augmentation via Claim Matching with LLMs
from load import df_fact_checks, df_posts, df_fact_check_post_mapping

import requests, json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

stop_words = list(stopwords.words('english'))+['list', 'extracted', 'key', 'entities']

# Need to download ollama from: https://ollama.com/download, to then download llama3.2:3b or any model (current is llama3.2:latest, which is the 3b model)
def Query_LLM_chat(model, prompt, parameters):    
    response = requests.post(
        "http://0.0.0.0:11434/api/chat",
        json={"model": model, "messages": prompt, "stream": False, "options": parameters}
    )

    try:
        response_text = response.text
        llm_response = json.loads(response_text)
        return llm_response['message']['content']
    except Exception as e:
        return("Error:", e)
    

def Entity_extract(post):
    #3-Shot Example (currently the examples aren't used as they were causing errors with the LLM responses)
    prompt = [
        {"role": "system", "content": "Extract the key entities (names, places, events, or concepts) from the Twitter post provided by the user. Only respond with entities that are contained in the Twitter post in a list format."},
        # {"role": "user", "content": f"NASA announces a new Artemis mission to land astronauts on the Moon by 2025, aiming to establish a sustainable lunar presence."},
        # {"role": "assistant", "content": f"NASA, Artemis, mission, astronauts, Moon, 2025, lunar"},
        # {"role": "user", "content": f"Tesla unveils its first electric semi-truck at a press event in Austin, Texas, aiming to revolutionize freight transportation."},
        # {"role": "assistant", "content": f"electric semi-truck, press event, Austin, Texas, freight transportation"},
        # {"role": "user", "content": f"Lionel Messi scores a stunning free-kick for Inter Miami in a thrilling match against LA Galaxy in the MLS. #GOAT"},
        # {"role": "assistant", "content": f"Lionel Messi, free-kick, Inter Miami, LA Galaxy, MLS, GOAT"},
        {"role": "user", "content": f"{post}"}
    ]

    parameters = {"temperature": 0.2, "max_tokens": 50, "top_p": 0.6}

    response = Query_LLM_chat("llama3.2:latest", prompt, parameters).lower()

    # if response from LLM, is not extracting entities but actually rejecting to answer due to it's guard rails. Then just use the post to find the entities
    if "can't" in response or "cannot" in response or "no entities" in response or "none" in response:
        response = post.lower()

    # Cleaning and filtering the responses to ensure only the key entities are returned
    clean_response = re.sub(r'[^\w\s]', '', response)

    filtered_response = ' '.join([w for w in clean_response.split() if w.lower() not in stop_words])

    entities = filtered_response.split()
    
    lem_entities = [lemmatizer.lemmatize(entity.lower()) for entity in entities]

    return lem_entities


# Was supposed to be used to filter for a specific language, but made accuracy worse
def Fact_filter(post_lang):
    possible_facts = []
    fact_id = 0 

    for fact in df_fact_checks["claim"]:
        # Cleaning the facts and adding the corresponding fact_id
        clean_fact = re.sub(r'[^\w\s]', '', fact[1].lower())
        possible_facts.append(f"fact_id {fact_id}: {clean_fact}")

        fact_id += 1

    return possible_facts


def Fact_search(possible_facts, entities):
    aligned_facts = []
    fact_points = []

    for possible_fact in possible_facts:
        points = 0
        for entity in entities:
            # If entity is found in fact
            if re.search(rf'\b{re.escape(entity)}s?\b', possible_fact, re.IGNORECASE):
                points += 1
        # Fact must have at least 1 common entity with the post to be used as evidence (threshold needs to be tested).
        if points >= 1:
            aligned_facts.append(possible_fact)
            fact_points.append(points)

    # If no facts align with the entities from the post, then all facts are used (need to work to find more efficient solution)
    if len(aligned_facts) == 0:
        aligned_facts = possible_facts
    
    return aligned_facts, fact_points


def Fact_decision(post_id, post, possible_facts, fact_points):
    # prompt the LLM
    facts = '- ' + '\n- '.join(possible_facts)

    prompt = [
        {"role": "system", "content": "Identify the fact that is most closely aligned with the Twitter post provided by the user. Only respond with the corresponding fact_id."},
        {"role": "user", "content": "post_id 1: NASA announces a new Artemis mission to land astronauts on the Moon by 2025, aiming to establish a sustainable lunar presence.\n\nThese are the related facts:\n1. fact_id: 10 - NASA announces a partnership with SpaceX for lunar lander development.\n2. fact_id: 20 - The Artemis program aims to establish a sustainable presence on the Moon by 2025.\n3. fact_id: 30 - NASA plans to launch the James Webb Space Telescope in 2021."},
        {"role": "assistant", "content": "fact_id: 20"},
        {"role": "user", "content": "post_id 2: Tesla unveils its first electric semi-truck at a press event in Austin, Texas, aiming to revolutionize freight transportation.\n\nThese are the related facts:\n1. fact_id: 15 - Tesla announces plans for a new gigafactory in Texas.\n2. fact_id: 25 - Tesla's electric semi-truck is designed to reduce emissions in freight transportation.\n3. fact_id: 35 - Tesla shares hit record highs after a successful quarter."},
        {"role": "assistant", "content": "fact_id: 25"},
        {"role": "user", "content": "post_id 3: Lionel Messi scores a stunning free-kick for Inter Miami in a thrilling match against LA Galaxy in the MLS. #GOAT\n\nThese are the related facts:\n1. fact_id: 40 - Lionel Messi makes his debut for Inter Miami in a friendly match.\n2. fact_id: 50 - Lionel Messi scores a match-winning free-kick for Inter Miami in the MLS.\n3. fact_id: 60 - Inter Miami announces a new signing from Barcelona."},
        {"role": "assistant", "content": "fact_id: 50"},
        {"role": "user", "content": f"post_id {post_id}: {post}.\n\nThese are the related facts:\n{facts}"}
    ]

    parameters = {"temperature": 0, "max_tokens": 50, "top_p": 0.6}

    response = Query_LLM_chat("llama3.2:latest", prompt, parameters)

    print(response)

    # Trying to find 'fact_id:' in the text to extract the id number
    match = re.search(r'fact_id:\s*(\d+)', response)

    # If able to find the 'fact_id:'
    if match:
        fact_id = match.group(1)
        return int(fact_id)
    else:
    # If unable to find the 'fact_id:'
        try:
            # Use the if from the fact which had the highest number of matching entities with the post
            highest_fact_point = max(fact_points)

            position = fact_points.index(highest_fact_point)

            fact = possible_facts[position]

            match = re.search(r'fact_id\s*(\d+)', fact) 
            fact_id = match.group(1)
            print("NEED EXTRA")
            return int(fact_id)
        except:
            print("No fact_id found in the response and no facts provided.")
            return -1


# Validating the answer, by finding the correct post to fact mapping in df_fact_check_post_mapping csv.
def valid_prediction(num_of_correct, post_id, fact_id):
    
    new_post_id = df_posts.index.to_list()[post_id]

    new_fact_id = df_fact_checks.index.to_list()[fact_id]

    pair_id = df_fact_check_post_mapping[df_fact_check_post_mapping['post_id'] == new_post_id].index.to_list()[0]

    actual_post_id = df_fact_check_post_mapping['post_id'].iloc[pair_id]
    actual_fact_id = df_fact_check_post_mapping['fact_check_id'].iloc[pair_id]

    # 1 point per correct answer, which is used to calculate the percentage accuracy at the end of the test.
    if actual_post_id == new_post_id and actual_fact_id == new_fact_id:
        num_of_correct += 1

    return num_of_correct, actual_post_id, actual_fact_id


# Main Script
num_of_correct = 0

for post_id in range(len(df_posts)):
    post = df_posts["text"].iloc[post_id]
    ocr = df_posts["ocr"].iloc[post_id]

    # Finding the valid textual information for each post, either in the text, ocr, or in both sections of the row.
    if post or ocr:
        if post and ocr:
            post_text = post[1] + ocr[0][1]
            post_lang = post[2][0][0] + ocr[0][2][0][0]
        elif post:
            post_text = post[1]
            post_lang = post[2][0][0]
        elif ocr:
            post_text = ocr[0][1]
            post_lang = ocr[0][2][0][0]

        # For Testing Purposes:
        # print(post_text)
        # print()

        entities = Entity_extract(post_text)
        # For Testing Purposes:
        # print(f"Entities: {entities}")
        # print()

        facts = Fact_filter(post_lang)

        relevant_facts, fact_points = Fact_search(facts, entities)
        # For Testing Purposes:
        # print("Amount of Relevant Facts:", len(relevant_facts))
        # print("Relevant Facts:",'- ' + '\n- '.join(relevant_facts))
        # print()

        fact_id = Fact_decision(post_id, post_text, relevant_facts, fact_points)

        print(f"Predict: {df_posts.index.to_list()[post_id]}, {df_fact_checks.index.to_list()[fact_id]}")
        num_of_correct, correct_post_id, correct_fact_id = valid_prediction(num_of_correct, post_id, fact_id)
        print(f" Actual: {correct_post_id}, {correct_fact_id}")
        print(f"Number of Correct Predictions: {num_of_correct}")

        print("----")

print(f"Accuracy: {(num_of_correct/(post_id+1))*100}%")

