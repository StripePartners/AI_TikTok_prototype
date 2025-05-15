import anthropic

client = anthropic.Anthropic(api_key="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")#os.getenv("ANTHROPIC_API_KEY"))

text = client.messages.create(
                                model="claude-3-5-sonnet-20240620",
                                max_tokens=400,
                                messages=[{"role":"user","content":"How are you?"}],
                                ) 
print(text.content[0].text)


