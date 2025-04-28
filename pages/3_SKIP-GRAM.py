import streamlit as st

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

# Set the parameters for the Word2Vec model
vector_size = 200
window_size = 10
min_count = 1
workers = 4
sg_flag = 1

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

# Preprocess the sentences
# Remove stopwords
tokenized_sentences = [simple_preprocess(remove_stopwords(sentence)) for sentence in sentences]

# Train a skip-gram Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=vector_size, window=window_size, min_count=min_count, workers=workers, sg=sg_flag)

# Get the vector for a word
word_vector = model.wv['russia']

# Get the most similar words
similar_words = model.wv.most_similar('russia')

st.title("SKIP-GRAM (With Preprocessing)")

st.write(f"Vector for 'russia': {word_vector}")

st.write(f"Most similar words to 'russia': {similar_words}")