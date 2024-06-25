import gradio as gr
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


general_prompt = """
Being an expert on {topic}, write a blog in {no_words} on {topic} in the language {language}, Give a suitable title, generate blog in points and always add author name as 'Rahul Gupta' at the end
"""

chat=ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="mixtral-8x7b-32768")



def blogger(topic,no_words,language,prt = general_prompt):

    prompt = PromptTemplate(input_variables = [topic,no_words,language],
                        template = prt)
    llm_chain=LLMChain(llm=chat,prompt=prompt)
    response = llm_chain.invoke({'topic':topic,'no_words':no_words,'language':language})
    return response['text']


topic = gr.Textbox(label="topic",placeholder="Like Machine Learning, Cooking, Football, Education in India",lines = 2)

no_words = gr.Textbox(label="no of words",placeholder="100")

language = gr.Textbox(label="language",placeholder="English, Hindi, Spanish, Tamil")

out= gr.Textbox(label="Output",placeholder="Output Blog will be returned here",lines = 25)

demo = gr.Interface(blogger,[topic,no_words,language],out,title="MixtraBlog AI: Fast and Multilingual Blog Generation Tool",description="This app is capable of generating Blogs with super high speeds and require minimum ram usage to generate Blog Post on any topic of your choice in specific language and number of words. Although the blogs generated in different or complex languages might not be accurate due to limited multilingual capabilities of my model !!")
demo.launch()