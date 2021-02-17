import torch

def computeTestSetAccuracy(data, model, loss_func, optimizer):
    '''
    Function to compute the accuracy on the test set
    Paramters
        :param model: Model to test
        :parar loss_func: 
        :param optimizer: Optimizer for computing gradients
    '''

    # Get DataLoader
    test_data_loader = data['test_dataloader']

    test_acc = 0.0
    test_loss = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        # set to evaluation mode
        model.eval()

        for j, (inputs, labels) in enumerate(test_data_loader):

            # inputs: 4D tensor (bs x 3 x width x height)
            inputs = inputs.to(device)
            # labels: 1D tensor (bs)
            labels = labels.to(device)

            # Forward pass
            # outputs: 2D tensor (batch_size x number_of_classes)
            outputs = model(inputs)

            # Loss
            loss = loss_func(outputs, labels)

            # Calculate loss and accuracy
            batch_size = inputs.size(0)
            test_loss += loss.item() * batch_size

            ret, predictions = torch.max(outputs.data, dim=1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            test_acc += acc.item() * batch_size

            print("Test Batch number: {:03d}, Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
    
    # Average loss and accuracy
    avg_test_loss = test_loss / len(test_data_loader.dataset)
    avg_test_acc = test_acc / len(test_data_loader.dataset)

    print("Test accuracy: {:.4f}%".format(avg_test_acc*100))