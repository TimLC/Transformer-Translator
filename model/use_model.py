import os
from timeit import default_timer as timer
from datetime import timedelta

import torch


class UseModel:
    PATH_MODEL = 'resources/model/'

    def __init__(self, model, name, device='cpu'):
        self.device = device
        self.name = name
        self.model = model.to(self.device)

    def train(self, optimizer, loss_fn, dataloader, numbers_epoch, load_model=True, save_model=True):
        if load_model:
            self.load_model()
        for epoch in range(1, numbers_epoch + 1):
            start_time = timer()
            train_loss = self.train_epoch(optimizer, loss_fn, dataloader)
            end_time = timer()
            val_loss = self.evaluate(loss_fn, dataloader)
            if save_model:
                self.save_model()
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"))

    def train_epoch(self, optimizer, loss_fn, dataloader):
        losses = 0
        cpt = 1
        self.model.train()
        for src, tgt in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:, :-1]

            logits = self.model(src, tgt_input)

            optimizer.zero_grad()

            tgt_out = tgt[:, 1:]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()
            print('Batch : {:.02%} | loss : {:.04f}'.format(cpt / len(dataloader), loss.item()))
            cpt += 1

        return losses / len(dataloader)

    def evaluate(self, loss_fn, dataloader):
        self.model.eval()
        losses = 0

        for src, tgt in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:, :-1]

            logits = self.model(src, tgt_input)

            tgt_out = tgt[:, 1:]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(dataloader)

    def save_model(self):
        if not os.path.exists(self.PATH_MODEL):
            os.makedirs(self.PATH_MODEL)
        torch.save(self.model, self.PATH_MODEL + self.name + '.pth')

    def load_model(self):
        if not os.path.isfile(self.PATH_MODEL + self.name + '.pth'):
            self.model = torch.load(self.PATH_MODEL + self.name + '.pth')