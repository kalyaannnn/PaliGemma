from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
            self, 
            hidden_size = 768, # Size of the embedding vector
            intermediate_size = 3072, # Size of the linear layer in the feedforward network
            num_hidden_layers = 12, # Number of layers of the vision transformer
            num_attention_heads = 12,
            num_channels = 3,
            image_size = 224,
            patch_size = 16, # Dimensions of the patch that the image will be divided into
            layer_norm_eps = 1e-6,
            attention_dropout = 0.0,
            num_image_tokens : int = None,
            **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbedding(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # Extracting information about the images patch by patch
        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim, # Equal to the embedding size, hidden layer
            kernel_size = self.patch_size, 
            stride = self.patch_size,
            padding = "valid", # Indicates that no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2 # Number of patches 224x224 / Patch_Size
        self.num_positions = self.num_positions # Number of positional encodings required which is equal to the number of patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent = False,
        )

    def forward(self, pixel_values : torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch Size, Channels, Height, Width]
        # Convolve the 'patch_size' kernel over the image, with no overlapping pathces since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch Size, Embed Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # [Batch Size, Embed Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids) # Add the positional encoding extracted from the embedding layer i.e from 0 - 15 to each patch
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings

class SiglipAttention(nn.Module):
    """ Multi Head Attention from paper - 'Attention is all you need' """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dims = self.embed_dim // self.num_heads
        self.scale = self.head_dims**-0.5 # Equivalent to 1/sqrt(self.hidden_dims)
        self.dropout = config.attention_droput

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj  = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj  = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states : torch.Tensor,) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Hidden State [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # Query States [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # Key States [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # Value States [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # Query States [Batch_Size,  Num_Patches, Num_Heads, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dims).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dims).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dims).transpose(1, 2)


        # Calculating the attention weights 
        # Q * K^T / sqrt(d_k) Attention Weights [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f"{attn_weights.size()}"
            )
        
        # Applying the softmax row wise, attention weights : [Batch Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float32).to(query_states.dtype)
        # Multiply the attention weights by the value states













class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states : torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # Hidden States : [Batch_size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate = "tanh") # Better in practice as it dictates the flow of gradient 
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config) # Self Attention Block
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)


    def forward(self, hidden_states : torch.Tensor) -> torch.Tensor:
        # Residual : [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states = hidden_states)
        # [Batch_size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # Residual : [Batch_Size, Num_Patches, Embed_Dim]
        resiudal = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states) # More degrees of freedom to learn for the model and prepare a sequence of batches to a next layer while adding non-linearity


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size # Size of the embedding vector

        self.embeddings = SiglipVisionEmbedding(config) # Extract the patches of images using this layer
        self.encoder = SiglipVisionEncoder(config) # Run it through a list of layers of transformers
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)

    def forward(self, pixel_values : torch.Tensor) -> torch.Tensor:
        # pixel_values : [Batch Size, Channels, Height, Width] -> [Batch Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(input_embeds = hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

class SiglipVisionModel(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_value) -> Tuple:
        # Vision Model will take a batch of images and give a batch of list of embeddings for each image where each image is of size Embed_Dim
        # [Batch Size, Channels, Height, Width] -> [Batch Size, Num Patches, Embed Dim]
        return self.vision_model(pixel_value = pixel_value)