from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np
import requests
from googlesearch import search
import re
from flask import Flask, render_template, request, redirect, url_for, flash
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/', methods=['POST'])
def get_company_name():
    if request.method == 'POST':
        company_name = request.form.get('company_name')
        if not company_name:
            flash('Please enter a company name.')
            return redirect(url_for('index'))
        else:
            # Process the company name
            return redirect(url_for('result', company_name=company_name))
    
    # If not a POST request, render the form
    return render_template('index.html')
    
def find_marketing_strategy(company_name):
    global final_summary
    final_summary=[]
    def scrape_data(company_name):
        """ just enter the company name and leave the rest to python :p 
        first we will get top 3 urls after searching for the company name's marketing strategy"""
        query = f"{company_name} marketing strategy"
        results = []
        final_output=[]
        for url in search(query, num_results=3):
            results.append(url)
        for url in results:
            page = requests.get(url)
            soup = BeautifulSoup(page.text, 'html.parser')
            content = soup.find('div')
            if content:
                para_in_div = content.find_all('p')
                para = [word.text.strip() for word in para_in_div]
                print(para)
                final_output.append(para)
            else:
                print(f"No content found for {url}")
        return final_output

    def clean_text(text):
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r"[[]] \n", " ", text)
        return text
    


    # Helper function to split text into chunks of ~1000 tokens
    def split_text_into_chunks(text, tokenizer, max_tokens=1024):
        words = text.split()
        chunks = []
        current_chunk = ""
        for word in words:
            # Tentatively add word
            temp_chunk = current_chunk + " " + word if current_chunk else word
            # Tokenize and check length
            tokenized = tokenizer(temp_chunk, return_tensors="pt", truncation=False)
            if tokenized['input_ids'].shape[1] <= max_tokens:
                current_chunk = temp_chunk
            else:
                chunks.append(current_chunk)
                current_chunk = word
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    # Process each chunk independently
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    def load_model_and_tokenizer(model_path="./bart-finetuned-marketing"):
        """
        Load the fine-tuned model and tokenizer from the specified path
        
        Args:
            model_path (str): Path to the saved model and tokenizer
            
        Returns:
            tuple: (model, tokenizer)
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        return model, tokenizer

    def generate_summary(text, model=None, tokenizer=None, model_path="./bart-finetuned-marketing", 
                        max_input_length=1024, max_output_length=256, num_beams=4):
        """
        Generate a summary for the given text using the fine-tuned BART model
        
        Args:
            text (str): Input text to summarize
            model: Pre-loaded model (if None, will load from model_path)
            tokenizer: Pre-loaded tokenizer (if None, will load from model_path)
            model_path (str): Path to the saved model and tokenizer (used if model or tokenizer is None)
            max_input_length (int): Maximum length for input text
            max_output_length (int): Maximum length for generated summary
            num_beams (int): Number of beams for beam search
            
        Returns:
            str: Generated summary
        """
        # Load model and tokenizer if not provided
        if model is None or tokenizer is None:
            model, tokenizer = load_model_and_tokenizer(model_path)
        
        # Ensure model is in evaluation mode
        model.eval()
        
        # Tokenize the input text
        inputs = tokenizer(text, max_length=max_input_length, truncation=True, padding="max_length",
                        return_tensors="pt")
        
        # Move to the same device as the model
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        # Generate summary
        summary_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_output_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    
    # Decode the generated summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        final_summary.append(summary)
        print(final_summary)
        return summary

    
    
    text = str(scrape_data(company_name))
    text=clean_text(text)
    print(f"clean data \n\n {text} \n\n")
    final_summary = []
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")  # or bart-large
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    # Your very long text
    long_text = text
    chunks = split_text_into_chunks(long_text, tokenizer)
    outputs = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=512, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        outputs.append(output_text)

    # Combine outputs
    final_output = "\n\n".join(outputs)

    print("Final Output:")
    print(final_output)
    final_summary.append(final_output)
    sample_text = text
    
    # Load model and tokenizer once (for efficiency when generating multiple summaries)
    model, tokenizer = load_model_and_tokenizer()
    
    # Generate summary
    summary = generate_summary(sample_text, model, tokenizer)
    
    print("Input Text:")
    print(sample_text[:100] + "..." if len(sample_text) > 100 else sample_text)
    print("\nGenerated Summary:")
    print(summary)
    return final_summary
    
def suggest(company_name, summary):
    cname=company_name
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate

    template = """
    You are a helpful assistant. Answer the question based on the context provided.
    based on company name and summary of competitor company, suggest me a better marketing strategy for my company
    Dont say/return anything like Since I don't have any specific information about your company or its products/services in the output
    context: {context}
    question: {question}
    Answer: 
    """

    model = OllamaLLM(model='llama3')
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    def conversation(company_name):
        question = f"suggest me better marketing strategy than my competitor {company_name} such that I can surpass them in market"
        context = summary
        result = chain.invoke({"context": context, "question":{question}}) 
        return result
    return conversation(cname)

def main():
    company_name = input("Enter the company name: ")
    output = find_marketing_strategy(company_name)
    suggestion = suggest(company_name, final_summary)
    print(f"\n\n\n Final strategy suggested is: \n \n  The current marketing strategy of {company_name} is \n {output}\n\n what you can do better is {suggestion}\n ")
    
if __name__ == "__main__":
    app.run(port=5000, debug=True)

