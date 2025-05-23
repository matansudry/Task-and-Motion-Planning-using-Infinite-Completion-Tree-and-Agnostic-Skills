import torch
from torch import nn

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(True), 
            nn.Linear(128, 512))#, 
            #nn.ReLU(True), 
            #nn.Linear(64, 128))
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        return x

class AttentaionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 12
        self.transformer = nn.Transformer(nhead=6, num_encoder_layers=12, d_model=dim)
        self.main_object_embedder = autoencoder()
        self.target_object_embedder = autoencoder()
        self.other_object_embedder = autoencoder()
        self.head = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.text_embedding_downsample = nn.Linear(512, 12)

    def forward(self, batch:dict):
        #extract state
        main_object = batch["state"][:,0,:]
        main_object = torch.unsqueeze(main_object, 1)
        target_object = batch["state"][:,1,:]
        target_object = torch.unsqueeze(target_object, 1)
        other_objects = batch["state"][:,2:,:]

        #create embedding
        """
        main_embedding = self.main_object_embedder(main_object)
        target_embedding = self.target_object_embedder(target_object)
        others_embedding = self.other_object_embedder(other_objects)
        text_embedding = self.text_embedding_downsample(batch["text_features"])
        text_embedding = torch.unsqueeze(text_embedding, 1)
        output = torch.cat((main_embedding, target_embedding, others_embedding, text_embedding), dim=1)
        """

        text_embedding = self.text_embedding_downsample(batch["text_features"])
        text_embedding = torch.unsqueeze(text_embedding, 1)
        output = torch.cat((batch['state'], text_embedding), dim=1)

        #run multihead
        output = self.transformer(output, output)[:,0,:]

        #run head
        output = self.head(torch.squeeze(output, 1))
        #output = torch.clamp(output, min=0, max=1)
        output = self.sigmoid(output)
        
        return output