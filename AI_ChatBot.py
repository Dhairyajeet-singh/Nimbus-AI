from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
You are a helpful assistant. Answer the question based on the context provided.
The context is also a conversation history.
Context: {context}
Question: {question}
Answer: 
"""

model = OllamaLLM(model='llama3')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def conversation():
    context = ""
    print("Welcome to AI Chatbot, Type exit to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":              #Exit EXit EXIT === exit
            break
        result = chain.invoke({"context": context, "question": user_input}) 
        print(f"Assistant: {result}")
        context += f"\n User: {user_input}\n Assistant: {result}\n"

if __name__ == "__main__":
    conversation()
