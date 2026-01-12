import torch, hydra
from PIL import Image
from transformers import DistilBertTokenizer
from model.model import CLIPModel
from torchvision import transforms
from hydra.utils import to_absolute_path

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # Load Model & Config
    device = cfg.system.device
    model = CLIPModel(cfg)
    model.clip.load_state_dict(torch.load(to_absolute_path("checkpoints/siglip_vietnamese_model.pt"), map_location=device))
    model.clip.eval()

    # Prepare Data
    image_path = "src/data/UIT-ViIC/dataset/test/images/000000007615.jpg"
    captions = ["Ở trên sân , một cầu thủ đánh bóng đang vung gậy để đánh bóng .", 
                "Người đàn ông cầm vợt tennis đang đuổi theo bóng .", 
                "Đứa trẻ đang đeo găng tay bóng chày bắt bóng thấp trên sân .", 
                "Hình ảnh kép của một số người đang chơi bóng đá trên sân ."]
    
    # Process & Inference
    img = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])(Image.open(image_path)).unsqueeze(0).to(device)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    inputs = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
    
    with torch.no_grad():
        logits = model.generate_similarity_matrix(img, inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
        probs = logits.softmax(dim=1).cpu().numpy()[0]

    # Print Result
    print("\nResult:")
    for text, p in sorted(zip(captions, probs), key=lambda x: x[1], reverse=True):
        print(f"{p*100:.2f}% : {text}")

if __name__ == "__main__":
    main()