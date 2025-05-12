# ☁️Nimbus-AI


Businesses often struggle to efficiently gather, analyze, and derive actionable insights from competitors' strategies and market movements. Additionally, startups and growing companies lack comprehensive intelligence on funding landscapes, investor preferences, and successful fundraising approaches. This information gap leads to missed opportunities, ineffective marketing strategies, and suboptimal business decisions.

**Nimbus-AI** is an AI-powered competitive intelligence platform that autonomously crawls and analyzes publicly available data on market competitors to extract actionable insights. **It benchmarks your competitors’ marketing strategies, funding patterns, investor relations, and business growth tactics, and then provides tailored recommendations to help your business surpass them.**

Nimbus AI leverages a blend of AI-driven web crawling, NLP, and data analytics to empower businesses with competitive intelligence and investment strategies previously accessible only to large corporations with dedicated market research teams.
The platform automates competitor monitoring, market analysis, and investor discovery, reducing manual research time and providing clear, actionable recommendations to outmaneuver rivals and secure growth funding.
By integrating an interactive dashboard and report generation system, Nimbus AI provides decision makers with:
Real-time competitor benchmarking reports.
Investor heatmaps showing active and interested VCs/angels in the sector.
Strategic action plans backed by data insights.
Ultimately, StrategiX levels the playing field, helping businesses outsmart, outpace, and outgrow their competitors.

# ⚒️Workflow

For this project, We have used flask to integrate a simple frontend with a complex backend. In further updates we will be enhancing the frontend. 
So just running the flask file will not work, since you would require to run data_and_model_creation.py file

**data_and_model_creation.py** file is basically a python file which allows us to fine-tune our model: the model we are using is BART. 
Since the overall working of this project is made in such a way that it's dependent on the fine-tuned bert model, that's why we just can't run the app.py file directly.
The fine-tuned model file was heavy in size, and was higher than the allowed memore to be shared over github that's why the model-fine-tuning script need to be run before the app.py

**so run the data_and_model_creation.py file for fintuned model and you are good to run the flask script. If you want to run the backend file go on to the Marketing_strategy.ipynb**

# ✨ Features

Text Summarization: Generate concise summaries from long documents or web content

Web Content Analysis: Extract and analyze content from URLs

Semantic Search: Find relevant information across your document library

Natural Language Processing: Perform sentiment analysis, named entity recognition, and more

LLM Integration: Leverage local LLMs via Ollama integration

Interactive UI: User-friendly interface for all functionality


# 🚀 Getting Started
  **Prerequisites**
  
    Python 3.8+
    pip package manager

  **Installation**
  
    bash
    git clone https://github.com/Dhairyajeet-singh/Nimbus-AI.git
    cd Nimbus-AI

  **Install the required dependencies:**

    bash
    pip install -r requirements.txt

  **Set up environment variables:**

    bash
    cp .env.example .env

  # Edit .env file with your API keys and configuration

  **Run the application:**

    bash
    # For Flask app
    python app.py

# 🧩 Architecture

**Nimbus-AI combines several components:**

Transformer Models: Utilizing Hugging Face's transformer library for state-of-the-art NLP tasks

Web Scraping: BeautifulSoup for extracting content from the web

Data Processing: Pandas and NumPy for structured data handling

LangChain Integration: For composable AI application development

Ollama: For local LLM integration

# 💡 Use Cases

Content Research: Quickly gather and analyze information from multiple sources

Document Analysis: Extract key information from large document collections

Text Enhancement: Improve writing through AI-powered suggestions

Data Insights: Visualize and interpret text-based data


