# %%
from docling.document_converter import DocumentConverter
# from docling.chunking import HybridChunker
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import uuid

#%% 
source = "DataArticles/exemplos.md"
converter = DocumentConverter()
result = converter.convert(source)
doc = result.document

# %%
# chunk_tokenizer = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
# chunker = HybridChunker(tokenizer=chunk_tokenizer, max_tokens=768, merge_peers=True)
# tokenizer = AutoTokenizer.from_pretrained(chunk_tokenizer)

embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

#%%

# Inicializa Qdrant em memória
qdrant = QdrantClient(":memory:")
collection_name = "documents_collection"
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

#%%
# Variáveis auxiliares
points = []
current_label = None

with open(source, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Detecta seção nova
        if line.startswith("# "):
            current_label = line[2:].strip().upper()  # Ex: FRAUD ou NO_FRAUD
            continue

        # Para linhas normais, cria embedding com o label atual
        if current_label is None:
            # Linha antes de definir seção, ignora
            continue

        embedding = embedding_model.encode(line).tolist()
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": line, "label": current_label}
        )
        points.append(point)

# Envia todos pontos para o Qdrant
qdrant.upload_points(
    collection_name=collection_name,
    points=points
)


# %%
query_raw = "Hello, I'm calling from Federal Revenue. There is a problem if your tax declarations, can you confirm some informations, plase?"
query_embedding = embedding_model.encode(query_raw).tolist()

result = qdrant.query_points(
    collection_name=collection_name,
    limit=3,
    query=query_embedding
)

labels = [p.payload["label"] for p in result.points]

print("Top 3 matched texts:")
for p in result.points:
    # Qdrant COSINE distance = 1 - cosine similarity
    cosine_similarity = 1 - p.score  # p.score é a distância COSINE retornada
    print(f"[{p.payload['label']}] similarity={cosine_similarity:.4f} | text: {p.payload['text']}")


fraud_votes = labels.count("FRAUD")
no_fraud_votes = labels.count("NO_FRAUD")

if fraud_votes > no_fraud_votes:
    print("\n>> This phrase is LIKELY FRAUD.")
else:
    print("\n>> This phrase is LIKELY NOT FRAUD.")
# %%

import joblib

# __define-ocg__ Salvando embeddings e informações associadas
texts = [p.payload["text"] for p in points]
labels = [p.payload["label"] for p in points]
vectors = [p.vector for p in points]

varOcg = {
    "texts": texts,
    "labels": labels,
    "embeddings": vectors,
    "model_name": 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
}

joblib.dump(varOcg, "modelo_fraude.joblib",compress=3)
print("Dados salvos com sucesso em 'modelo_fraude.joblib'")

# %%
import zipfile

with zipfile.ZipFile("modelo_fraude.zip", "w") as zipf:
    zipf.write("modelo_fraude.joblib")

# %%
