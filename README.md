# Outiftter-AI
Outfitter.ai transforms outfit selection with an AI-driven platform that blends personal  style with real-time weather data, delivering stylish, practical recommendations. 
Starting with 1,700 outfits, the system curates 475 complete ensembles, each with 
core pieces like tops and trousers for functionality. OpenAI’s CLIP model converts 
clothing images into 512-dimensional embeddings, averaged to capture an outfit’s 
visual and semantic essence, enabling precise style matching. 

Weather integration ensures comfort and suitability—a sharp suit fails in rain, and a t
shirt falters in cold. The OpenWeatherMap API converts local temperature and 
precipitation into labels like “rainy,” “cold,” “hot,” or “normal,” encoded via CLIP and 
merged with user inputs. Manchester’s unpredictable weather, shifting from sunny 
mornings to gusty showers by noon, underscores this need, ensuring outfits adapt to 
sudden changes and avoid wardrobe mishaps. 
Personalised Recommendations 

The system excels in two modes: 

• **Text Input**: Enter a style (e.g., “business casual”) and your city. Outfitter.ai 
combines this with the weather label—say, “business casual for cold weather”—and 
uses FAISS to retrieve the top three outfit matches. 

• **Image Input**: Upload a clothing photo, and CLIP encodes it. This embedding 
merges with the weather vector, enabling FAISS to find outfits that reflect your style 
and suit the forecast. 

A user-friendly Gradio interface, with tabs for text and image inputs, presents results 
in an elegant gallery, simplifying the experience. 

**Opportunities for Growth**:

Despite its strengths, challenges include basic weather labels omitting humidity, 
equal weighting of outfit items diluting key pieces like coats, and limited 
personalisation. Future plans involve fine-tuning CLIP for enhanced fashion-weather 
alignment, incorporating detailed weather metrics (e.g., humidity, wind), prioritising 
core items, adding user profiles for tailored suggestions, and expanding data via 
augmentation. Outfitter.ai is poised to lead in personalised, weather-savvy fashion 
innovation. 
