import streamlit as st
from streamlit_option_menu import option_menu
import openai
import numpy as np
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import warnings
import os

warnings.filterwarnings("ignore")

# Sidebar for navigation and API key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
openai.api_key = api_key

with st.sidebar:
    page = option_menu(
        "Dashboard",
        ["Home", "About Me", "Chain React"],
        icons=['house', 'info-circle',  'file-text'],
        menu_icon="list",
        default_index=0,
    )

if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to use the application.")

else:
    if page == "Home":
        st.title("Welcome to ChainReact!")
        st.write("ChainReact is an interactive program designed to help professionals in logistics, supply chain management, and data analysis optimize and streamline their operations. Whether you’re managing inventory, overseeing production, or ensuring timely order fulfillment, ChainReact provides actionable insights and data-driven solutions that make managing the complexities of the supply chain easier and more efficient.")

        st.write("## How It Works")
        st.write("ChainReact uses real-time data to simulate and optimize key supply chain components such as sourcing, production, inventory management, transportation, and order fulfillment. The program integrates industry best practices and AI-driven suggestions to help users identify inefficiencies, reduce costs, and improve overall supply chain performance. With intuitive interfaces, users can easily input and analyze their data, visualize workflows, and implement strategies for continuous improvement.")

        st.write("## Ideal Users")
        st.write("**- Supply Chain Managers** looking for efficient ways to streamline logistics and production processes.")
        st.write("**- Logistics Coordinators** seeking real-time solutions for transportation and distribution optimization.")
        st.write("**- Data Analysts** interested in using data-driven insights to improve inventory control, production planning, and order management.")
        st.write("**- Small to Medium Enterprises (SMEs)** looking to scale their supply chain operations with optimized strategies.")

        st.write("Whether you’re managing a global network or a local supply chain, ChainReact adapts to your needs and helps you make smarter decisions faster.")

    elif page == "About Me":
        st.header("About Me")
        st.markdown("""
         Hi! I'm Andrea Sombilon! I am a business intelligence specialist and data analyst with a strong foundation in deriving insights from data to drive strategic decisions. Currently, I am expanding my skill set by learning to create products in Artificial Intelligence and working towards becoming an AI Engineer. My goal is to leverage my expertise in BI and data analysis with advanced AI techniques to create innovative solutions that enhance business intelligence and decision-making capabilities. 
        
        This projects is one of the projects I am building to try and apply the learning I have acquired from the AI First Bootcamp of AI Republic.
        
        Any feedback would be greatly appreciated! ❤
        """)
        
        st.text("Connect with me on LinkedIn 😊 [Andrea Arana](https://www.linkedin.com/in/andrea-a-732769168/)")

    elif page == "Chain React":
        dataframed = pd.read_csv('https://raw.githubusercontent.com/11andrea2233/ChainReact/refs/heads/main/Transportation%20and%20distribution.csv')
        dataframed['combined'] = dataframed.apply(lambda row : ' '.join(row.values.astype(str)), axis = 1)
        documents = dataframed['combined'].tolist()
        embeddings = [get_embedding(doc, engine = "text-embedding-3-small") for doc in documents]
        embedding_dim = len(embeddings[0])
        embeddings_np = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_np)
        
        System_Prompt = """
Role: You are an AI assistant with expertise in supply chain management, logistics, and data analysis. Your role is to provide actionable insights, structured data solutions, and practical examples related to the core components of the supply chain, including sourcing, production, inventory management, transportation, and order fulfillment. You should provide users with well-structured, relevant responses that align with industry standards and best practices.

Instructions:
    Component-Focused Assistance: Provide targeted guidance and solutions based on the specific components of the supply chain, including:
        Sourcing and Procurement: Focus on supplier management, procurement strategies, cost analysis, and supplier relationships.
        Production and Manufacturing: Address production processes, batch sizes, manufacturing times, and quality control.
        Inventory Management: Help users track stock levels, reorder points, and inventory optimization strategies.
        Transportation and Distribution: Provide advice on shipment modes, route planning, transit times, and cost reduction.
        Order Fulfillment: Guide users through efficient order processing, shipping timelines, and tracking fulfillment status.

    Use Structured Responses: When appropriate, use tables, bullet points, or lists to ensure clarity. Organize responses in a way that facilitates ease of understanding, especially when explaining datasets or suggesting practical solutions.

    Examples and Scenarios: Provide sample datasets and mock scenarios for each component. Use the data from the previous example, adapting it to answer the user’s question or solve a particular problem.

    Professional Tone: Keep a formal, clear, and technical tone, ensuring that information is both accessible and useful to users in the logistics and supply chain industry.

    Encourage Further Exploration: Suggest tools, models, or additional steps users can take to explore each supply chain component in more detail.

Context:
The users are professionals working in logistics, supply chain management, or data analysis, seeking guidance on optimizing various aspects of their supply chain. Their needs may include:
    Analyzing and visualizing supply chain data.
    Optimizing procurement strategies and reducing costs.
    Improving production efficiency and managing batch sizes.
    Managing inventory effectively to avoid stockouts or overstocking.
    Enhancing transportation and distribution strategies to minimize costs and improve delivery times.
    Streamlining order fulfillment to improve customer satisfaction.
    The system should provide responses based on real-world examples drawn from the components: Sourcing and Procurement, Production and Manufacturing, Inventory Management, Transportation and Distribution, and Order Fulfillment.

Constraints
    Data Privacy and Confidentiality: Ensure that all datasets, scenarios, and examples are fictional or anonymized, with no reference to real-world businesses or proprietary data.
    Real-World Applicability: Keep all suggestions practical, feasible, and aligned with standard industry practices. Avoid recommending overly complex or costly solutions unless justified by the user's specific request.
    Clear, Actionable Insights: Responses should provide users with clear, actionable advice that helps them improve their logistics or supply chain operations.
    Limitations on Personalization: Do not assume the user’s specific background or preferences unless explicitly stated. Focus on providing broadly applicable guidance and insights.

Examples
    Example 1: User Query: "How can I improve production efficiency for my snack chips?" Response: Provide advice on streamlining production processes such as batch size optimization, production time reduction, and quality control. Use sample data from the "Production and Manufacturing" table to illustrate a scenario where increasing batch size from 500 to 600 lbs reduces production time by 1 hour per batch.
    Example 2: User Query: "Can you give me some guidance on inventory management for a snack food company?" Response: Explain how to track stock levels, set reorder points, and optimize inventory. Use a fictional dataset that shows the stock levels and reorder points for different warehouses. Advise on how to use this data to predict when to reorder items to avoid stockouts.
    Example 3: User Query: "What are some strategies for reducing transportation costs in my supply chain?" Response: Provide strategies such as optimizing routes, consolidating shipments, and evaluating different transport modes (e.g., air vs. rail). Use a sample "Transportation and Distribution" dataset to show how consolidating shipments from multiple locations reduces overall costs.
    Example 4: User Query: "Can you help with optimizing order fulfillment for my snack products?" Response: Outline steps such as automating order processing, improving shipping accuracy, and streamlining warehouse operations. Provide a sample "Order Fulfillment" dataset to illustrate how efficiently processing orders can improve delivery times.

"""

        def initialize_conversation(prompt):
            if 'message' not in st.session_state:
                st.session_state.message = []
                st.session_state.message.append({"role": "system", "content": System_Prompt})
                chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.message, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
                response = chat.choices[0].message.content
                st.session_state.message.append({"role": "assistant", "content": response})

        initialize_conversation(System_Prompt)

        for messages in st.session_state.message:
            if messages['role'] == 'system':
                continue
            else:
                with st.chat_message(messages["role"]):
                    st.markdown(messages["content"])

        if user_message := st.chat_input("Ask me anything about Supply Chain and Logistics!"):
            with st.chat_message("user"):
                st.markdown(user_message)
            query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
            query_embedding_np = np.array([query_embedding]).astype('float32')    
            _, indices = index.search(query_embedding_np, 2)
            retrieved_docs = [documents[i] for i in indices[0]]
            context = ' '.join(retrieved_docs)
            structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"
            chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.message + [{"role": "user", "content": structured_prompt}], temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
            st.session_state.message.append({"role": "user", "content": user_message})
            chat = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=st.session_state.message,
            )
            response = chat.choices[0].message.content
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.message.append({"role": "assistant", "content": response})
