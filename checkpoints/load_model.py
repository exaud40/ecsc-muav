import torch

def load_best_model(model, path, device):

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    print(f"Best accuracy: {checkpoint['accuracy']:.4f}")

    model.eval()

    return model


    


