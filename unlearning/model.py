import torch
import numpy as np

class TrainMethods():
    def train_model(model,loss_function,optimizer,train_loader,num_epochs=9):

        for epoch in range(num_epochs):
            model.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    def train_model_gradients(model,train_dataloader,loss_function,optimizer,threshold,num_epochs=6):
        for epoch in range(num_epochs):
            model.train()
            for data, target in train_dataloader:
                optimizer.zero_grad()
                outputs = model(data)
                loss = loss_function(outputs, target)
                loss.backward()
                
                for name, param in model.named_parameters():
                    if param.grad is not None and param.requires_grad and param.grad.abs().mean() <= threshold:
                        param.requires_grad = False
                
                optimizer.step()
            
            print(f'Epoch {epoch+1}/{6}, Loss: {loss.item()}')

    def train_model_noise(model, train_dataloader,optimizer,loss,epsilon,num_epochs=6):
        for epoch in range(num_epochs):
            model.train()
            for data, target in train_dataloader:
                optimizer.zero_grad()
                output = model(data)
                loss = loss(output, target)
                loss.backward()
                
                # Add noise to gradients
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.tensor(np.random.laplace(loc=0, scale=1/epsilon, size=param.grad.shape))
                        param.grad += noise
                
                optimizer.step()
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

class EvaluationMethods():
    def eval_model_conf(test_dataloader,model,loss):
        model.eval()
        all_targets = []
        all_predictions = []
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                output = model(data)
                test_loss += loss(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                pred = output.argmax(dim=1)
                all_targets.extend(target.tolist())
                all_predictions.extend(pred.tolist())

        test_loss /= len(test_dataloader.dataset)
        accuracy = 100. * correct / len(test_dataloader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)} ({accuracy:.0f}%)\n')

        return all_targets, all_predictions
    
    def eval_model(model,test_dataloader,loss):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                output = model(data)
                test_loss += loss(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_dataloader.dataset)
        accuracy = 100. * correct / len(test_dataloader.dataset)

        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)} '
        f'({accuracy:.0f}%)\n')


class SaveMethods():
    def save_model(model,name):
        torch.save(model.state_dict(), 'model/'+name+'.pth')
        print('Model saved to '+name+'.pth')
