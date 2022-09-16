import torch
import torch.nn as nn
import torch.nn.functional as F

# Neural Network Model (1 hidden layer)
class Text_CNN(nn.Module):
    def __init__(self, filter_num, f_in, emb_dim, num_posts, dropout):
        super(Text_CNN, self).__init__()

        self.hidden_size = emb_dim

        ### TEXT CNN
        channel_in = 1
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, f_in)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        ## Class  Classifier
        self.class_classifier = nn.Sequential(nn.Linear( self.hidden_size, 2),
                                              nn.Softmax(dim=1))


    def forward(self, text, A):

        ##########CNN##################
        text = text.unsqueeze(dim=1)# add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]
        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = F.leaky_relu(self.fc1(text))

        ### Fake or real
        class_output = self.class_classifier(text)

        return class_output, text


class Reconstruction(nn.Module):
    def __init__(self, filter_num,  emb_dim, f_in, num_posts, window_size, dropout):
        super(Reconstruction, self).__init__()

        self.num_posts = num_posts
        self.fc = nn.Linear(emb_dim, len(window_size) * filter_num, bias= False)
        ### TEXT CNN
        channel_in = 1
        self.window_size = window_size
        self.dconvs = nn.ModuleList([nn.ConvTranspose2d(filter_num, channel_in,  (K, f_in)) for K in self.window_size])

    def forward(self, x):
        x = self.fc(x)
        x_rec = list(x.chunk(len(self.window_size),dim=1))
        size = [self.num_posts - self.window_size[i] + 1 for i in range(len(self.window_size))]
        text = [F.interpolate(x_rec[i].unsqueeze(dim=2), size[i], mode='linear') for i in range(len(self.window_size))]
        text = [F.leaky_relu(self.dconvs[i](text[i].unsqueeze(3))) for i in range(len(self.dconvs))]
        text = torch.cat(text, dim=1)
        text = torch.mean(text, dim =1).squeeze(dim=1)

        return text


class Text_CNN_CCFD(nn.Module):
    def __init__(self, filter_num, f_in, emb_dim, num_posts, dropout):
        super(Text_CNN_CCFD, self).__init__()
        self.hidden_size = emb_dim

        ### TEXT CNN
        channel_in = 1
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, f_in)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.reconstruction = Reconstruction(filter_num, emb_dim, f_in, num_posts, window_size, dropout)

        self.domaingate = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                        nn.Sigmoid())

        self.classifier_all = nn.Linear(emb_dim, 2)
        self.classifier_com = nn.Linear(emb_dim, 2)

    def gateFeature(self, x, gate):
        xc = torch.mul(x, gate)
        xs = torch.mul(x, 1 - gate)
        return xc, xs

    def forward(self, text, A):
        ##########CNN##################
        text = text.unsqueeze(dim=1)  # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]
        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text_copy = text.copy()
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        xf = F.leaky_relu(self.fc1(text))

        dgate = self.domaingate(xf)
        xc, xs = self.gateFeature(xf, dgate)

        xall = xc + xs
        preds = self.classifier_all(xall)

        preds_xc = self.classifier_com(xc)
        preds_xs = self.classifier_all(xc.detach() + xs)

        x_rec = self.reconstruction(xc + xs)

        return preds, preds_xc, preds_xs, xc, xs, x_rec, dgate
