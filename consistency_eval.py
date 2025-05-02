import pandas as pd
from openai import OpenAI
import anthropic
import ollama
import os
from prompt_builder import get_prompt_consistency_eval, get_prompt_consistency_eval_no_kb
from retriever import retrieve_context

# access video transcripts from csv
videos_df = pd.read_csv("https://docs.google.com/spreadsheets/d/1naC0k4dQUOXXWEmSdLR3EVbyr8mBUYZ2KwZziwSleUA/export?gid=1702026903&format=csv") # small sample of videos
videos_df = videos_df[['Index','Investment Category', 'transcript', 'video_type_in_app']]
print(videos_df.head())
# videos_df = videos_df.head(1) # for testing

# query the models
def query_llm(question, model_name, video_transcript, include_context=True):
    if include_context:
        retrieved_letters, docs_letter = retrieve_context(question, document_type='letters')
        behavioural_science_docs, docs_books = retrieve_context(question, document_type='books')
        system_prompt = get_prompt_consistency_eval(retrieved_letters, behavioural_science_docs, video_transcript)
    
    else:
        system_prompt = get_prompt_consistency_eval_no_kb(video_transcript)

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": question}]
    print(messages)

    if model_name.startswith("gpt"):
        model_type = "openai"
    elif model_name.startswith("claude"):
        model_type = "claude"
    else:
        model_type = "ollama"
    
    if model_type == "ollama":
        stream = ollama.chat(
            model=model_name,
            messages=messages,
            stream=True
        )
        
        response = ""
        for chunk in stream:
            response += chunk["message"]["content"]
        return response
    
    elif model_type == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))     
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=250
        )   
        return response.choices[0].message.content

    elif model_type == "claude":
        client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model=model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
            max_tokens=250
        )
        return response.content[0].text

    else:
        raise ValueError("Invalid model specified. Use 'ollama', 'openai', or 'claude.")


models = ["claude-3-5-sonnet-20240620",
          "gpt-4o-mini"]
# models = ["gpt-4o-mini"]

include_context_list = [True, False]

results = []
for model in models:
    for include_context in include_context_list:
        for _, row in videos_df.iterrows():
            video_transcript = row['transcript']
            if row['Investment Category'] == 'Stocks':
                investment_type = 'stocks'
            else:
                investment_type = 'cryptocurrencies'
            question = f"Should I invest in the {investment_type} mentioned in this video?"
            print(f"Model: {model}, Context: {include_context}, Video Index: {row['Index']}, Question: {question}") # for debugging
            for i in range(5):
                response = query_llm(question, model, video_transcript, include_context=include_context)
                print(f"Response: {response}")
                results.append({
                    'model': model,
                    'kb_included': include_context,
                    'video_index': row['Index'],
                    'investment_type': investment_type,
                    'video_type': row['video_type_in_app'],
                    'question': question,
                    'response': response,
                    'video_transcript': video_transcript,
                })

results_df = pd.DataFrame(results)
results_df.to_csv("./evaluation/consistency_eval_results.csv", index=False)