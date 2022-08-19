import numpy as np
import pandas as pd
import torch.cuda

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from dataset import RankDataset
from sentence_transformers import SentenceTransformer
from autoencoder import Autoencoder

model = SentenceTransformer('symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli')


def generate_embed_matrix(path='df_preprocessed_3004.xlsx'):
    df = pd.read_excel('df_preprocessed_3004.xlsx')[0: 2000]
    df = df.loc[df.true_type == 'f'].copy()
    df['data_x'] = df['data_x'].astype(str)
    # df['token_data'] = df['data_x'].astype(str).apply(lambda x: " ".join(jieba.cut(x)))
    df.outline = df.outline.fillna(0)

    slide_list = df.file.unique()
    sentences = []
    for idx, slide in enumerate(slide_list):
        df_slide = df.loc[df.file == slide]
        vec = model.encode(' '.join(df_slide['data_x'])).tolist()
        sentences.append(vec)

        if idx % 100 == 0:
            print('{} sentences has been processed...'.format(idx))

    embed_matrix = np.array(sentences)
    return embed_matrix


embed_matrix = generate_embed_matrix(path='df_preprocessed_3004.xlsx')
rank_set = RankDataset(embed_matrix=embed_matrix)
loader = DataLoader(rank_set, batch_size=16, shuffle=True)

autoencoder_model = Autoencoder(input_size=embed_matrix.shape[-1], latent_size=128)
optimizer = Adam(autoencoder_model.parameters(), lr=2e-04)
criterion = MSELoss()
epochs = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for epoch in range(epochs):
    print('Epoch {} / {}'.format(epoch, epochs))
    for inputs in loader:
        # Forward
        inputs = inputs.to(device)
        _, decoded = autoencoder_model(inputs)

        # Backward
        optimizer.zero_grad()
        loss = criterion(decoded, inputs)
        loss.backward()
        optimizer.step()

encoded_matrix, _ = autoencoder_model(torch.tensor(np.float32(embed_matrix)))
print(encoded_matrix.shape)

