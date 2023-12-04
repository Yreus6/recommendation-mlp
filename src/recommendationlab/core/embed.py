from torch import nn


class UserEmbedding(nn.Module):
    def __init__(self, num_users: int, emb_size: int):
        super(UserEmbedding, self).__init__()
        self.user_embedding = nn.Embedding(num_users, emb_size)

    def forward(self, users):
        return self.user_embedding(users)


class ItemEmbedding(nn.Module):
    def __init__(self, num_items, emb_size: int):
        super(ItemEmbedding, self).__init__()
        self.item_embedding = nn.Embedding(num_items, emb_size)

    def forward(self, items):
        return self.item_embedding(items)
