import torch
from torch import nn
from torch import optim

from transformer.utils import DEVICE

def adjust_learning_rate(optimizer, d_model, step_num, warmup_steps=4000):

    lr = d_model**(-0.5) * min(step_num**(-0.5), step_num * warmup_steps**(-1.5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, dataset, params, n_epochs=5, print_every=100, 
          save_model=True):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_plot = []
    
#     for epoch in range(1, n_epochs + 1):
    epoch = 0
    while True:
        epoch += 1
        
        prev_mean_epoch_loss = 0
        mean_epoch_loss = 0
        
#         for batch_idx, (inputs, targets) in enumerate(dataloader):
        for batch_idx, pair in enumerate(dataset):
        
            inputs = torch.LongTensor(pair['input']).unsqueeze(-1).to(device)
            targets = torch.LongTensor(pair['target']).unsqueeze(-1).to(device)

            optimizer.zero_grad()
            
            outputs, prediction = model(inputs, targets)
            
            # when we use teacher forcing, output and target are not same dim
            if len(outputs) > len(targets):
                # padding with zeros (find better way)
                new_targets = torch.ones(len(outputs), targets.shape[1], dtype=torch.long).to(device)
                
                new_targets[:len(targets)] = targets
                targets = new_targets
            elif len(outputs) < len(targets):
                # dont know how else to do for now
                targets = targets[:len(outputs)]

            loss = criterion(outputs.view(-1, outputs.shape[-1]), targets[:len(outputs)].view(-1))
            
            loss.backward()
            optimizer.step()
            
            mean_epoch_loss += loss.item()
            
            if (batch_idx + 1) % print_every == 0:
#             if True:
            
                print("Batch %d / %d ----- Loss : %.4f" % (batch_idx, len(dataset), loss.item()/len(targets)))
                print('Q : ' + ''.join(id2word[str(pred.item())] + ' ' for pred in inputs.squeeze()))
                print('A : ' + ''.join(id2word[str(pred.item())] + ' ' for pred in prediction.squeeze()))
                
                
                loss_plot.append(mean_epoch_loss - prev_mean_epoch_loss)
                prev_mean_epoch_loss = mean_epoch_loss
#                 break
            
        plt.plot(np.array(loss_plot) / print_every, 'r')
        plt.xlabel('number of examples')
        plt.ylabel('Loss')
        plt.savefig('loss.png')
        plt.clf
        np.savetxt('loss.csv', np.array(loss_plot) / print_every)
            
        mean_epoch_loss /= len(dataset)
        print("Epoch %d ----- Loss : %.4f" % (epoch, mean_epoch_loss))
        
        if save_model:
            torch.save(model.state_dict(), 'trained_model.pt')
            
            
            
if __name__ == "__main__":
    
    from preprocessing import loadDataset
    import numpy as np

    pairs, vocab_size, word2id, id2word = loadDataset()
    np.random.shuffle(pairs)
    
    model = Seq2Seq(input_size=vocab_size,
                    embedding_size=256,
                    hidden_size=256,
                    num_layers=5,
                    teacher_forcing_prob=0.5).to(device)
    train(model, pairs)