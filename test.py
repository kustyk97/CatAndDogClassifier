# imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import argparse
from model import ImageClassifier
import os

# Set image size.
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 1

MODEL_STATE_DICT_KEY = "model_state_dict"


# parse arguments
def get_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path to the ImageClassifier model."
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset folder.")
    return parser.parse_args()


def load_model(model_path: str) -> ImageClassifier:
    """Load the trained model from a checkpoint."""
    model = ImageClassifier().to(DEVICE)
    try:
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint[MODEL_STATE_DICT_KEY])
        # classes_names = checkpoint[CLASSES_KEY]
        return model
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")


def get_dataloader(dataset_path: str) -> DataLoader:
    """Create a DataLoader for the dataset."""
    data_transform = transforms.Compose(
        [transforms.Resize(size=IMAGE_SIZE), transforms.ToTensor()]
    )
    dataset = datasets.ImageFolder(root=dataset_path, transform=data_transform)

    dataloader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )
    return dataloader


def test(
    model: ImageClassifier,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
) -> float:
    """Evaluate the model on the test dataset."""

    model.eval()

    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(DEVICE), y.to(DEVICE)

            pred_logits = model(X)

            loss = loss_fn(pred_logits, y)
            test_loss += loss.item()

            pred_labels = pred_logits.argmax(dim=1)
            test_acc = (pred_labels == y).sum().item() / len(pred_labels)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def log_results(dataset_path: str, model_path: str, loss: float, acc: float) -> None:
    """Log results to a text file."""
    with open("results.txt", "a") as f:
        f.write(f"Path to testset: {dataset_path}\n")
        f.write(f"Tested model path: {model_path}\n")
        f.write(f"Loss score: {loss}\n")
        f.write(f"Accuracy: {acc}\n")


def main():

    args = get_args()
    model_path = args.model
    dataset_path = args.dataset
    # load model and dataset
    model = load_model(model_path)
    test_dataloader = get_dataloader(dataset_path)

    # test model
    loss_fn = torch.nn.CrossEntropyLoss()
    loss, acc = test(model, test_dataloader, loss_fn)
    # print result
    print(f"Loss score: {loss}")
    print(f"Accuracy: {acc}")
    log_results(dataset_path, model_path, loss, acc)


if __name__ == "__main__":
    main()
