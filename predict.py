import torch
from PIL import Image
from main import image_transforms

def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''

    test_image = Image.open(test_image_name)
    test_image_tensor = image_transforms['test'](test_image)
    test_image_tensor - test_image_tensor.view(1, 3, 244, 244)

    with torch.no_grad():
        
        # set to evaluation mode
        model.eval

        # Forward pass
        # output: 1D tensor (number_of_classes)
        output = model(test_image_tensor)

        # exponential of output: class probablities
        ps = torch.exp(output)

        # topk: k largest values alon dim
        topk, topclass = ps.topk(3, dim=1)
        predicted_label = topclass.cpu().numpy()[0][0]
        score = topk.cpu().numpy()[0][0]

    for i in range(3):
        print("Predcition", i+1, ":", topclass.cpu().numpy()[0][i], ", Score: ", topk.cpu().numpy()[0][i])

    print('output', output)
    print('ps', ps)
    print('topk', topk)
    print('topclass', topclass)
    print('topclass.cpu()', topclass.cpu())
    print('topclass.cpu().numpy()', topclass.cpu().numpy())
    print('topk.cpu()', topk.cpu())
    print('topk.cpu().numpy()', topk.cpu().numpy())