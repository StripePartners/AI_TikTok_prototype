import pandas as pd
import ollama
from openai import OpenAI
import anthropic
import os
from prompt_builder import get_prompt_eval, get_prompt_eval_no_kb
from retriever import retrieve_context

# access list of factual probes from csv
df = pd.read_csv('./evaluation/generated_questions.csv')
df = df[df['relevant_question'] == 'yes']
columns = ['question', 'answer']
df = df[columns].reset_index(drop=True)
print(df.head())
# df = df.head(2) # for testing

# input into LLM (start with ollama)
def query_llm(question, model_name="llama3:8b", include_context=True):
    if include_context:
        retrieved_letters, docs_letter = retrieve_context(question, document_type='letters')
        behavioural_science_docs, docs_books = retrieve_context(question, document_type='books')
        system_prompt = get_prompt_eval(retrieved_letters, behavioural_science_docs)
    
    else:
        system_prompt = get_prompt_eval_no_kb()

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
            model=model_name, # "gpt-4o-mini" or "gpt-4o",
            messages=messages,
            max_tokens=150
        )   
        return response.choices[0].message.content

    elif model_type == "claude":
        client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model=model_name, # "claude-3-5-sonnet-20240620", "claude-3-5-haiku-20241022"
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
            max_tokens=150
        )
        return response.content[0].text

    else:
        raise ValueError("Invalid model specified. Use 'ollama', 'openai', or 'claude.")

# anthropic_models = [
#     "claude-3-5-sonnet-20240620",
#     "claude-3-5-haiku-20241022"
# ]

# openai_models = [
#     "gpt-4o", 
#     "gpt-4o-mini"
# ]

results = []
for _, row in df.iterrows():
    question = row['question']
    gold_answer = row['answer']
    
    # set to include_context to False to exclude KB. Set model_name to desired OpenAI/Claude/Ollama model.
    llm_response = query_llm(question, model_name="gpt-4o")
    print(f"LLM Response: {llm_response}")
    
    results.append({"question": question, "gold_answer": gold_answer, "llm_response": llm_response})

results_df = pd.DataFrame(results)
results_df.to_csv('./evaluation/factual_probe/results/gpt_4o.csv', index=False)