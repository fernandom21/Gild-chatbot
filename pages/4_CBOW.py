import streamlit as st

import plotly.express as px
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

# Set the parameters for the Word2Vec model
vector_size = 200
window_size = 10
min_count = 1
workers = 4
sg_flag_ngram = 1
sg_flag_cbow = 0

# Sample sentences
sentences = [
    "President Zelenskyy addresses the UN General Assembly in person for the first time since Russia began its invasion of his country",
    "Alexander Lukashenko suggests Belarus could revive an old alliance with Russia and North Korea.",
    "Rustem Umerov is appointed Ukraine’s defence minister, replacing Oleksii Reznikov",
    "Kim Jong-un and Vladimir Putin hold a summit in Russia to discuss a possible deal to supply North Korean arms for the war in Ukraine",
    "Zelenskyy warns other parts of Europe would be at risk from Russia’s military aggression if it succeeded in the war in Ukraine",
    "Ukraine’s Minister of Defence announces Sweden has approved a €270m security assistance package for Ukraine",
    "President Zelenskyy says Ukraine is not slowing down the pace of its ambitions to join NATO",
    "President Zelenskyy dismisses Ukraine’s ambassador to London, Vadym Prystaiko",
    "The NATO Secretary-General outlines a three-part, multi-year package that will bring Ukraine closer to NATO",
    "Putin says he will do everything to protect Russia"
]

# Preprocess the sentences to remove stopwords
# Remove stopwords
tokenized_sentences = [simple_preprocess(remove_stopwords(sentence)) for sentence in sentences]

# Train a skip-gram Word2Vec model
skip_gram_model = Word2Vec(tokenized_sentences, vector_size=vector_size, window=window_size, min_count=min_count, workers=workers, sg=sg_flag_ngram)

# Train a CBOW Word2Vec model
cbow_model = Word2Vec(tokenized_sentences, vector_size=vector_size, window=window_size, min_count=min_count, workers=workers, sg=sg_flag_cbow)

# Get the vector for the word "russia" in each model
sg_cat_vector = skip_gram_model.wv['russia']
cbow_cat_vector = cbow_model.wv['russia']

# Get the most similar words to "russia" in each model
sg_similar_words = skip_gram_model.wv.most_similar('russia')
cbow_similar_words = cbow_model.wv.most_similar('russia')

import pandas as pd

# Create a dataframe of vectors for the words "russia" and "sweden" in each model
df = pd.DataFrame({
    'Model': ['SKIP-gram', 'CBOW', 'SKIP-gram', 'CBOW'],
    'Word': ['russia', 'russia', 'sweden', 'sweden'],
    'X': [skip_gram_model.wv['russia'][0], cbow_model.wv['russia'][0], skip_gram_model.wv['sweden'][0], cbow_model.wv['sweden'][0]],
    'Y': [skip_gram_model.wv['russia'][1], cbow_model.wv['russia'][1], skip_gram_model.wv['sweden'][1], cbow_model.wv['sweden'][1]]
})

# Create a scatter plot of the vectors, with different colors for each model
fig = px.scatter(df, x='X', y='Y', color='Model', hover_name='Word', size_max=38)

# Set the plot title and axis labels
fig.update_layout(
    xaxis_title="X",
    yaxis_title="Y",
    title="Visualization of Skip-gram and CBOW Word Embeddings"
)

# ⭐ Display in Streamlit
st.title("CBOW")
st.plotly_chart(fig)