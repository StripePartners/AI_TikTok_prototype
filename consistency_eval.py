import pandas as pd
from openai import OpenAI
import anthropic
import ollama
import os
from prompt_builder import get_prompt_consistency_eval, get_prompt_consistency_eval_no_kb, get_prompt_eval_strategy, get_prompt_eval_strategy_no_kb
from retriever import retrieve_context


# query the models
def query_llm(question, model_name, video_transcript, include_context=True, eval_type="consistency"):
    if include_context:
        retrieved_letters, docs_letter = retrieve_context(question, document_type='letters')
        behavioural_science_docs, docs_books = retrieve_context(question, document_type='books')
        if eval_type == "consistency":
            system_prompt = get_prompt_consistency_eval(retrieved_letters, behavioural_science_docs, video_transcript)
        else:
            system_prompt = get_prompt_eval_strategy(retrieved_letters, behavioural_science_docs, video_transcript)
    
    else:
        if eval_type == "consistency":
            system_prompt = get_prompt_consistency_eval_no_kb(video_transcript)
        else:
            system_prompt = get_prompt_eval_strategy_no_kb(video_transcript)

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

# evaluation of consistency of responses
def consistency_eval(models, include_context_list, videos_df):
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
                    response = query_llm(question, model, video_transcript, include_context=include_context, eval_type="consistency")
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
    return results_df

# evaluation of strategy identification
def strategy_eval(models, include_context_list, videos_df):
    results = []
    for model in models:
        for include_context in include_context_list:
            for _, row in videos_df.iterrows():
                video_transcript = row['transcript']
                question = f"""
                Which of the following strategies is used in this video: FOMO, overconfidence, or authority bias? \n
                If a mix of strategies is used in the video, list them all. \n
                Only reference the three strategies listed above. \n
                Provide a short justification for your answer. \n
                """
                print(f"Model: {model}, Context: {include_context}, Video Index: {row['Index']}, Question: {question}") # for debugging
                response = query_llm(question, model, video_transcript, include_context=include_context, eval_type="strategy")
                print(f"Response: {response}")
                results.append({
                    'model': model,
                    'kb_included': include_context,
                    'video_index': row['Index'],
                    'video_type': row['video_type_in_app'],
                    'question': question,
                    'response': response,
                    'video_transcript': video_transcript,
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv("./evaluation/strategy_eval_results.csv", index=False)
    return results_df


models = ["claude-3-5-sonnet-20240620",
          "gpt-4o-mini"]
# models = ["gpt-4o-mini"]

# include_context_list = [True, False]
include_context_list = [True]

# access video transcripts from csv
videos_df = pd.read_csv("https://docs.google.com/spreadsheets/d/1naC0k4dQUOXXWEmSdLR3EVbyr8mBUYZ2KwZziwSleUA/export?gid=1702026903&format=csv") # small sample of videos
videos_df = videos_df[['Index','Investment Category', 'transcript', 'video_type_in_app']]
print(videos_df.head())
# videos_df = videos_df.head(1) # for testing

consistency_results_df = consistency_eval(models, include_context_list, videos_df)
print(consistency_results_df.head())
# strategy_results_df = strategy_eval(models, include_context_list, videos_df)
# print(strategy_results_df.head())