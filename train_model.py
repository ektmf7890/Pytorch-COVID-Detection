import torch

def train_and_validate(data, train_data_size, valid_data_size, model, loss_func, optimizer, epochs=25):
    '''
    Parmaeters
        :param model: Model to train and validate
        :param loss_func: Loss function to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs
    Returns
        model: Trained model
        best_epoch: returns the index of the epoch with best accuracy
        history: dict object, Training loss, accuracy and validation loss, accuracy
    '''

    # Get Data Loaders
    train_dataloader = data['train_dataloader']
    valid_dataloader = data['valid_dataloader']

    # Epoch: Train + Validation
    # Train: Forward pass -> Back propagation(get gradient) -> Update parameters -> Loss, Accuracy
    # Validation: Forward pass -> Loss, Accuracy

    best_loss = 100000
    best_epoch = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        # set the model to training mode
        model.train()

        # Loss and accurarcy for this epoch
        train_loss = 0
        train_acc = 0
        valid_loss = 0
        valid_acc = 0


        for i, (inputs, labels) in enumerate(train_dataloader):
            # inputs: 4D tensor (batch_size x 3 x width x height)
            inputs = inputs.to(device)
            # labels: 1D tensor (batch_size)
            labels = labels.to(device)

            # Clear existing gradients
            optimizer.zero_grad()

            # Forward pass
            # outputs: 2D tensor (batch_size x number_of_classes)
            outputs = model(inputs)

            # Loss
            loss = loss_func(outputs, labels)

            # Backward pass: calculate gradients for parameters
            loss.backward()

            # Update parameters
            optimizer.step()

            # Calculate loss and accuracy
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            
            # predictions: 1D tensor (batch_size), class index with the largest probablility of every image in the batch
            ret, predictions = torch.max(outputs.data, 1)

            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * batch_size

            '''
            print("inputs", inputs)
            print("inputs.size(0)", inputs.size(0))
            print("outputs", outputs)
            print("predictions", predictions)
            print("labels", labels)
            print("labels.data.view_as(predictions)", labels.data.view_as(predictions))
            print("loss", loss)
            print("los.item()", loss.item())
            print("acc", acc)
            print("acc.item()", acc.item())
            '''

            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))
            
        # Validation: No gradient checking needed
        with torch.no_grad():

            # set to evaluation mode
            model.eval()

            for j, (inputs, labels) in enumerate(valid_dataloader):
                # inputs: 4D tensor (bs x 3 x width x height)
                inputs = inputs.to(device)
                # labels: 1D tensor (bs)
                labels = labels.to(device)

                # Clear existing gradients
                optimizer.zero_grad()

                # Forward pass
                # outputs: 2D tensor (batch_size x number_of_classes)
                outputs = model(inputs)

                # Loss
                loss = loss_func(outputs, labels)

                # Calculate loss and accuracy
                batch_size = inputs.size(0)
                valid_loss += loss.item() * batch_size

                ret, predictions = torch.max(outputs.data, dim=1)
                # view(shape of output), view_as(tensor whose shape is to be mimicked)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * batch_size
        
        # Average loss and accuracy of this epoch
        # i+1: number of batches in train set, j+1: number of batches in valid set
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        # Save the model is it has the bect valid_acc until now
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            best_epoch = epoch
            torch.save(model, 'COVID19'+'_model_'+str(epoch)+'.pt')

        print("Epoch{:02d}: training loss {:.4f}, training accuracy {:.4f}%, validation loss {:.4f}, validation accuracy {:.4f}%".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100))
        
    return model, best_epoch