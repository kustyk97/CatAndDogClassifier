import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
from model import ImageClassifier
import argparse

# Set image size.
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_STATE_DICT_KEY = "model_state_dict"
CLASSES_KEY = "classes"


def get_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path to the ImageClassifier model."
    )
    parser.add_argument("--image", required=True, help="Path to the image file.")
    return parser.parse_args()


def load_image(image_path: str) -> (torch.tensor, Image.Image):
    """Load and transform an image."""
    transform = transforms.Compose(
        [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]
    )

    try:
        image = Image.open(image_path)
        return transform(image).unsqueeze(0), image
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")


def load_model(model_path: str) -> (ImageClassifier, list):

    model = ImageClassifier().to(DEVICE)
    try:
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint[MODEL_STATE_DICT_KEY])
        classes_names = checkpoint[CLASSES_KEY]
        return model, classes_names
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")


def predict(
    model: ImageClassifier, image_tensor: torch.tensor, class_names: list
) -> str:
    """Make predictions on the input image."""
    model.eval()
    with torch.no_grad():
        pred = model(image_tensor.to(DEVICE))
        pred_probs = torch.softmax(pred, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1)
        class_name = class_names[pred_label.cpu()]
        return class_name


def display_image(image: Image.Image, class_name: str):
    """Display the image with its predicted class name."""
    plt.figure(figsize=(10, 7))
    plt.imshow(image)
    plt.axis("off")
    plt.title(class_name, fontsize=14)
    plt.show()


def main():

    # parse arg
    args = get_args()
    image_path = args.image
    model_path = args.model

    # load image
    image_tensor, image_orginal = load_image(image_path)

    # load model
    model, class_names = load_model(model_path)

    # predict
    class_name = predict(model, image_tensor, class_names)

    # print results
    print(f"Predicted class: {class_name}")
    # below code don't work inside containder
    display_image(image_orginal, class_name)


if __name__ == "__main__":
    main()
