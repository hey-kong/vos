import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPProcessor, CLIPModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_ID = "clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_ID)
preprocess = CLIPImageProcessor.from_pretrained(model_ID)
processor = CLIPProcessor.from_pretrained(model_ID)
model = model.to(device)


# Define a function to load an image and preprocess it for CLIP
def load_and_preprocess_image(image_path):
    # Load the image from the specified path
    image = Image.open(image_path)
    # Apply the CLIP preprocessing to the image
    image = preprocess(image, return_tensors="pt")
    # Return the preprocessed image
    return image


def clip_img_features(img_path):
    image = load_and_preprocess_image(img_path)["pixel_values"]
    image_features = model.get_image_features(image.cuda())
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features


def clip_text_features(test_labels):
    text_inputs = processor([f"This is a photo of {c}" for c in test_labels], padding=True, return_tensors="pt")
    # Calculate the embeddings for the images using the CLIP model
    text_features = model.get_text_features(input_ids=text_inputs['input_ids'].cuda(),
                                            attention_mask=text_inputs['attention_mask'].cuda()).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


if __name__ == "__main__":
    test_labels = ['dog', 'cat', 'pig']  # text must be a list
    text_features = clip_text_features(test_labels)

    img_path = 'test/test1.jpg'
    img_features = clip_img_features(img_path)

    # output = torch.nn.functional.cosine_similarity(img_features, text_features)
    output = img_features @ text_features.T
    print(output.tolist()[0])
