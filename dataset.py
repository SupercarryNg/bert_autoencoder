import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer


class RankDataset(Dataset):
    def __init__(self, embed_matrix):
        super(RankDataset, self).__init__()
        self.embed_matrix = embed_matrix

    def __getitem__(self, idx):
        return torch.from_numpy(self.embed_matrix[idx]).float()

    def __len__(self):
        return self.embed_matrix.shape[0]


if __name__ == '__main__':
    sentences = ['现在我想做翻个好人', '好啊同法官讲啊', '对不起我是差人', '看下边个信你']
    model = SentenceTransformer('symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli')
    embed_matrix = model.encode(sentences)
    rank_set = RankDataset(embed_matrix)
    print(embed_matrix.shape)
    # print(next(iter(rank_set)))
