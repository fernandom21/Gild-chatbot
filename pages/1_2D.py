import streamlit as st

import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

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
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# Train a Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get the word vectors
word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])

# Reduce the dimensions to 3D using PCA
pca = PCA(n_components=3)
reduced_vectors = pca.fit_transform(word_vectors)

# Color setup
color_map = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'purple',
    4: 'orange',
    5: 'cyan',
    6: 'magenta',
    7: 'yellow',
    8: 'pink',
    9: 'brown'
}

word_colors = []
for word in model.wv.index_to_key:
    for i, sentence in enumerate(tokenized_sentences):
        if word in sentence:
            word_colors.append(color_map[i])
            break

word_ids = [f"word-{i}" for i in range(len(model.wv.index_to_key))]

scatter = go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers+text',
    text=model.wv.index_to_key,
    textposition='top center',
    marker=dict(color=word_colors, size=8),
    customdata=word_colors,
    ids=word_ids,
    hovertemplate="Word: %{text}<br>Color: %{customdata}"
)

# Create line traces for each displayed sentence
display_array = [True, True, False, False, False, True, True, True, False, False]  # Example display array

# Create line traces for each sentence
line_traces = []
for i, sentence in enumerate(tokenized_sentences):
    if display_array[i]:
        line_vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in sentence]
        line_trace = go.Scatter(
            x=[vector[0] for vector in line_vectors],
            y=[vector[1] for vector in line_vectors],
            mode='lines',
            line=dict(color=color_map[i], width=1, dash='solid'),
            showlegend=True,
            name=f"Sentence {i+1}",  # Customize the legend text
            hoverinfo='all'  # Disable line trace hover info
        )
        # Set different marker symbols for the start and end words
        line_traces.append(line_trace)

fig = go.Figure(data=[scatter] + line_traces)

# Set the plot title and axis labels
fig.update_layout(
    xaxis_title="X",
    yaxis_title="Y",
    title="2D Visualization of Word Embeddings",
    width=1000,  # Custom width
    height=1000  # Custom height
)

# ⭐ Display in Streamlit
st.title("Word2Vec - 2D")
st.plotly_chart(fig)