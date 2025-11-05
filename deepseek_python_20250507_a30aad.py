import numpy as np
import torch
import faiss
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
import requests
import gradio as gr

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Model Ensemble Setup
# -----------------------------
class FashionEnsemble:
    def __init__(self):
        # BLIP-2 for detailed understanding
        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        # CLIP for style similarity
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        
        # FAISS index for fashion knowledge
        self.faiss_index = faiss.IndexFlatIP(512)  # CLIP embedding size
        self._init_knowledge_base()
        
        # Weather API
        self.weather_api_key = "your_openweather_api_key"
        
    def _init_knowledge_base(self):
        """Preload with fashion items from a knowledge base"""
        # In production, this would connect to a fashion database
        sample_items = [
            {"description": "white linen shirt", "embedding": np.random.rand(512)},
            {"description": "black leather jacket", "embedding": np.random.rand(512)},
            # ... more items
        ]
        for item in sample_items:
            self._add_to_index(item)

    def _add_to_index(self, item):
        """Add item to FAISS index"""
        emb = np.array(item["embedding"]).astype('float32').reshape(1, -1)
        self.faiss_index.add(emb)

# -----------------------------
# Enhanced Weather Integration
# -----------------------------
class WeatherAwareRecommender:
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.weather_rules = {
            'cold': {'materials': ['wool', 'cashmere'], 'layers': 2},
            'hot': {'materials': ['linen', 'cotton'], 'colors': ['light']},
            'rainy': {'waterproof': True}
        }
    
    def get_weather_context(self, city):
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.ensemble.weather_api_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            
            if response.status_code != 200:
                return None
                
            return {
                'temp': data['main']['temp'],
                'condition': data['weather'][0]['main'].lower(),
                'humidity': data['main']['humidity']
            }
        except:
            return None
    
    def weather_appropriate(self, outfit_desc, weather):
        """Score outfit based on weather rules"""
        if not weather:
            return 0.5  # Neutral score if no weather data
            
        condition = weather['condition']
        rules = self.weather_rules.get(condition, {})
        
        score = 0
        for key, values in rules.items():
            if isinstance(values, list):
                score += any(v in outfit_desc.lower() for v in values)
            elif values is True:
                score += int(key in outfit_desc.lower())
                
        return min(1.0, score * 0.3)  # Max 30% weight for weather

# -----------------------------
# Knowledge-Enhanced Recommendation
# -----------------------------
class KnowledgeAugmenter:
    def __init__(self, ensemble):
        self.ensemble = ensemble
    
    def find_similar_items(self, query_embedding, k=3):
        """Find similar items from knowledge base"""
        query_emb = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.ensemble.faiss_index.search(query_emb, k)
        return [(i, d) for i, d in zip(indices[0], distances[0])]
    
    def augment_with_knowledge(self, user_items):
        """Enhance user's items with similar items from knowledge base"""
        augmented = []
        
        for item in user_items:
            # Get CLIP embedding
            inputs = self.ensemble.clip_processor(
                text=item['description'], 
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                emb = self.ensemble.clip_model.get_text_features(**inputs).cpu().numpy()
            
            # Find similar items
            similar = self.find_similar_items(emb)
            for idx, score in similar:
                if score > 0.6:  # Similarity threshold
                    augmented.append({
                        'source': 'knowledge',
                        'description': f"Similar to your {item['description']}: {self.get_knowledge_item(idx)}",
                        'score': score
                    })
        
        return user_items + augmented[:3]  # Add top 3 similar items

# -----------------------------
# Complete Recommendation System
# -----------------------------
class FashionRLAgent:
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.reward_model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-4)
        
    def update(self, outfit_embedding, reward):
        """Update reward model with user feedback"""
        x = torch.FloatTensor(outfit_embedding).unsqueeze(0).to(device)
        y = torch.FloatTensor([reward]).to(device)
        
        pred = self.reward_model(x)
        loss = nn.MSELoss()(pred, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class FashionRecommender:
    def __init__(self):
        self.ensemble = FashionEnsemble()
        self.weather = WeatherAwareRecommender(self.ensemble)
        self.knowledge = KnowledgeAugmenter(self.ensemble)
        self.rl_agent = FashionRLAgent(self.ensemble)
        
    def recommend(self, user_prompt, user_items, city=None):
        # 1. Augment with knowledge base
        enhanced_items = self.knowledge.augment_with_knowledge(user_items)
        
        # 2. Get weather context
        weather = self.weather.get_weather_context(city)
        
        # 3. Generate recommendations
        outfits = self._generate_outfits(user_prompt, enhanced_items, weather)
        
        # 4. Score with RL
        scored = []
        for outfit in outfits:
            emb = self._get_outfit_embedding(outfit['description'])
            with torch.no_grad():
                rl_score = self.rl_agent.reward_model(torch.FloatTensor(emb).to(device)).item()
            weather_score = self.weather.weather_appropriate(outfit['description'], weather)
            
            outfit['score'] = 0.6 * rl_score + 0.3 * weather_score + 0.1 * outfit.get('similarity', 0.5)
            scored.append(outfit)
        
        return sorted(scored, key=lambda x: -x['score'])[:3]

# -----------------------------
# Gradio Interface
# -----------------------------
recommender = FashionRecommender()

with gr.Blocks() as demo:
    gr.Markdown("## 👗 Advanced Fashion Recommender")
    
    with gr.Row():
        with gr.Column():
            user_id = gr.Textbox(label="User ID")
            prompt = gr.Textbox(label="Outfit Request")
            city = gr.Textbox(label="City (for weather)")
            recommend_btn = gr.Button("Get Recommendations")
            
        with gr.Column():
            output = gr.Textbox(label="Recommendations", interactive=False)
            feedback = gr.Slider(1, 5, label="Rate this recommendation")
            feedback_btn = gr.Button("Submit Feedback")
    
    recommend_btn.click(
        lambda p, c: recommender.recommend(p, [], c),
        inputs=[prompt, city],
        outputs=output
    )
    
    feedback_btn.click(
        lambda fid, f: recommender.rl_agent.update_last(fid, f),
        inputs=[user_id, feedback],
        outputs=output
    )

demo.launch()