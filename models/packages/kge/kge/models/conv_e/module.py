import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = input_tensor.size()
        return input_tensor.view(batch_size, -1)


class ConvEModule(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_height: int = 20,
        embedding_width: int = 10,
        conv_out_channels: int = 32,
        conv_kernel_size: int = 3,
        embedding_dropout: float = 0.2,
        feature_map_dropout: float = 0.2,
        projection_dropout: float = 0.3,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_height = embedding_height
        self.embedding_width = embedding_width

        embedding_size = embedding_height * embedding_width
        flattened_size = (
            (embedding_width * 2 - conv_kernel_size + 1)
            * (embedding_height - conv_kernel_size + 1)
            * conv_out_channels
        )

        self.entity_embeddings = nn.Embedding(
            num_embeddings=self.num_entities, embedding_dim=embedding_size
        )
        self.relation_embeddings = nn.Embedding(
            num_embeddings=self.num_relations, embedding_dim=embedding_size
        )

        self.convolution_pipeline = nn.Sequential(
            nn.Dropout(p=embedding_dropout),
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_out_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=conv_out_channels),
            nn.Dropout2d(p=feature_map_dropout),
            Flatten(),
            nn.Linear(in_features=flattened_size, out_features=embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=embedding_size),
            nn.Dropout(p=projection_dropout),
        )

    def forward_conv(
        self, subject_indices: torch.Tensor, relation_indices: torch.Tensor
    ) -> torch.Tensor:
        subject_embeddings = self.entity_embeddings(subject_indices)
        relation_embeddings = self.relation_embeddings(relation_indices)

        subject_embeddings = subject_embeddings.view(
            -1, self.embedding_width, self.embedding_height
        )
        relation_embeddings = relation_embeddings.view(
            -1, self.embedding_width, self.embedding_height
        )
        conv_input = torch.cat([subject_embeddings, relation_embeddings], dim=1).unsqueeze(1)
        conv_output: torch.Tensor = self.convolution_pipeline(conv_input)

        return conv_output

    def forward(
        self,
        subject_indices: torch.Tensor,
        relation_indices: torch.Tensor,
        object_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conv_output = self.forward_conv(
            subject_indices=subject_indices, relation_indices=relation_indices
        )

        if object_indices is not None:
            object_embeddings = self.entity_embeddings(object_indices)
            scores = torch.sum(conv_output * object_embeddings, dim=1, keepdim=True)
        else:
            scores = conv_output.mm(self.entity_embeddings.weight.t())

        return scores

    def get_entity_embeddings(self, entity_indices: torch.Tensor) -> torch.Tensor:
        embeddings: torch.Tensor = self.entity_embeddings(entity_indices)
        return embeddings

    def get_relation_embeddings(self, relation_indices: torch.Tensor) -> torch.Tensor:
        embeddings: torch.Tensor = self.relation_embeddings(relation_indices)
        return embeddings
