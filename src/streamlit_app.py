import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os

# Interface do Streamlit
st.set_page_config(page_title="Fraud Storage",page_icon="🕵‍♀️")
st.title("Fraud Phrase Detection App")

"""
# Welcome!!

### This is a fraud detection app. 

If you received any suspected sms, let us know and we can scam for you. 

Use `/fraud-detection-ia-app.py` to see if your message is suspected

If you have any questions, checkout our [documentation](https://github.com/amandapaura/Gen-IA-App).

"""
num_check = st.slider("Number of check you whant to make", 3, 5, 1)

# ---------------------------

@st.cache_resource  # ou st.cache(allow_output_mutation=True) se estiver com versão mais antiga
def load_model():
    if not os.path.exists("modelo_fraude.joblib"):
        with zipfile.ZipFile("modelo_fraude.zip", "r") as zip_ref:
            zip_ref.extractall()
    modelo = joblib.load("modelo_fraude.joblib")
    return modelo

data = load_model()


query = st.text_area("Enter a phrase to classify:", height=100)

if st.button("Analyze"):
    if query.strip() == "":
        st.warning("Please enter a phrase.")
    else:
        # Carrega o dicionário salvo        
        texts = data["texts"]
        labels = data["labels"]
        embeddings = data["embeddings"]
        model_name = data["model_name"]

        # Inicializa o modelo de embedding (recarrega a partir do nome salvo)
        embedding_model = SentenceTransformer(model_name)

        # Inicializa Qdrant em memória e cria coleção
        qdrant = QdrantClient(":memory:")
        collection_name = "documents_collection"

        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

        # Cria os pontos a partir dos dados salvos (texts, labels, embeddings)
        points = []
        import uuid
        for i in range(len(texts)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={"text": texts[i], "label": labels[i]}
            )
            points.append(point)

        # Envia os pontos para Qdrant
        qdrant.upload_points(collection_name=collection_name, points=points)


        query_embedding = embedding_model.encode(query).tolist()

        result = qdrant.query_points(
            collection_name="documents_collection",
            limit=num_check,
            query=query_embedding
        )

        labels = [p.payload["label"] for p in result.points]

        exp1 = st.expander('Results')
        
        fraud_votes = labels.count("FRAUD")
        no_fraud_votes = labels.count("NO_FRAUD")

        if fraud_votes > no_fraud_votes:
            exp1.success("🔴 This phrase is **LIKELY FRAUD**.")
        else:
            exp1.info("🟢 This phrase is **LIKELY NOT FRAUD**.")
    
        # ------------------------------------------
        exp2 = st.expander(f'Top {num_check} most similar texts')
        for i, p in enumerate(result.points):
            cosine_similarity = 1 - p.score  # distância cosine invertida
            exp2.markdown(f"""
                **#{i+1}**  
                Label: `{p.payload['label']}`  
                Cosine Similarity: `{cosine_similarity:.4f}`  
                Text: _{p.payload['text']}_
                """)

        # -----------------------------------
        exp3 = st.expander('Registred data')
        tab_data, tab_plot = exp3.tabs(['Database','WordCloud'])

        df_reg = pd.read_csv("DataArticles/dados.csv")

        with tab_data:
            st.dataframe(df_reg)

        with tab_plot:
            fig, ax = plt.subplots()
            text = " ".join(review for review in df_reg['sentence'])
            print(text)
            stopwords = set(STOPWORDS)
            wordcloud = WordCloud(stopwords=stopwords, 
                            background_color="white", 
                            max_words=100, 
                            max_font_size=50).generate(text)
            # Display the generated image:
            ax.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)