import streamlit as st
import requests
 # HNSW stands for Hierarchical Navigable Small World — it a graph-based algorithm used for fast
# Approximate Nearest Neighbor (ANN) search, which is exactly what you need in a RAG pipeline with FAISS.
# convert to FAISS index manually
st.title("Ecommerce AI Assistant")
 
if "chat" not in st.session_state:
    st.session_state.chat = []
 
user_input = st.text_input("Message")
 
if st.button("Send"):
    response = requests.post("http://localhost:8000/chat", params={"query": user_input})
    answer = response.json()["answer"]
 
    st.session_state.chat.append(("user", user_input))
    st.session_state.chat.append(("bot", answer))
 
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"🟢 **You:** {msg}")
    else:
        st.markdown(f"⚪ **Bot:** {msg}")
 