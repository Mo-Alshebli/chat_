from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
# from utils import *
import pinecone
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

model = ["gpt-3.5-turbo",
         "gpt-3.5-turbo-16k-0613",

         ]
st.subheader("روبوت محادثة اعتماد على بيانات سابق باستحدام تقنية الذكاء الاصطناعي")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["مرحبا كيف يمكنني مساعدتك ؟"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []


if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

llm = ChatOpenAI(model_name=model[1],
                 openai_api_key="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")

system_msg_template = SystemMessagePromptTemplate.from_template(template="""أجب على السؤال بأكبر قدر ممكن من الصدق 
باستخدام السياق المتوفر ، وأريد الإجابة كلها باللغة العربية وإذا لم تكن الإجابة موجودة في النص أدناه ، قل "لا أعرف 
واذا تم سؤال من انت قول انا روبوت دردشة معتمد على بيانات سابقة تم تدريبي عليها  "'""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages \
        (
        [
            system_msg_template,
            MessagesPlaceholder(variable_name="history"),
            human_msg_template
        ]
    )

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)
# Set up your OpenAI API key
load_dotenv()
pinecone.init(api_key="eb457224-95f8-4881-a32d-b28bcf8adb23", environment="us-west4-gcp-free")

# Define the name for your Pinecone index
index_name = 'fir'


# Embed the query using GPT-3.5-turbo


def find_match(input, model):
    embeddings = OpenAIEmbeddings(model=model)
    embedding = embeddings.embed_query(input)
    pinecone_index = pinecone.Index(index_name)
    response = pinecone_index.query(
        vector=embedding,
        top_k=2,
        # include_values=True,
        includeMetadata=True
    )
    return response['matches'][0]['metadata']['text'] + "\n" + response['matches'][1]['metadata']['text']


def query_refiner(conversation, query, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": f"بالنظر إلى استعلام المستخدم وسجل المحادثة قم بإعادة صياغة السؤال الى افضل طريقة  هذا السؤال "
                        f"{query}:"
                        f" والنص السابق التابع للمحادثة هو { conversation}"
                        f"قم بضياغة السؤال بحيث يكون سهل ومن نفس البيانات "},
        ],
        temperature=0.7,
        max_tokens=232,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    refined_query = response.choices[0].message.content

    return refined_query


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string
# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if st.button("Process"):
        if query:
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                # refined_query = query_refiner(conversation_string, query,model[0])
                # st.subheader("Refined Query:")
                # st.write(refined_query)
                context = find_match(query, model[1])
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], is_user=True, key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], key=str(i) + '_user')






