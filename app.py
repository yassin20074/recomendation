import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading ML and embedding")
 
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

df_places= pd.read_csv("C:/Users/matrg/Desktop/recomendation/pleace.csv")
 
places_embeddings = embedding_model.encode(df_places['description'].tolist(), show_progress_bar=False)


def generate_timeline(matched_places, budget):
    """خوارزمية تقسيم الوقت وصناعة الـ Itinerary ليتوافق مع الـ UI بالصور الحقيقية"""
    program = []
    current_time = 9.0   
    total_spent = 0.0
    visited_types = set()

    
    place_images = {
        1: "https://images.unsplash.com/photo-1539650116574-8efeb43e2750?w=500&q=80",
        2: "https://images.unsplash.com/photo-1553913861-c0fddf2619ee?w=500&q=80",
        3: "https://images.unsplash.com/photo-1605649487212-47bdab064df7?w=500&q=80",
        4: "https://images.unsplash.com/photo-1541518763669-27fef04b14ea?w=500&q=80",
        5: "https://images.unsplash.com/photo-1554118811-1e0d58224f24?w=500&q=80",
        6: "https://images.unsplash.com/photo-1603360946369-dc9bb6258143?w=500&q=80",
        7: "https://images.unsplash.com/photo-1590075865003-e48277faf551?w=500&q=80",
        8: "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80",
        9: "https://images.unsplash.com/photo-1568772585407-9361f9bf3a87?w=500&q=80",
        10: "https://images.unsplash.com/photo-1596123000693-01037617c092?w=500&q=80",
        11: "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=500&q=80",
        12: "https://images.unsplash.com/photo-1501443712940-3de752531a80?w=500&q=80",
        13: "https://images.unsplash.com/photo-1572252009286-268acec5ca0a?w=500&q=80",
        14: "https://images.unsplash.com/photo-1544025162-d76694265947?w=500&q=80",
        15: "https://images.unsplash.com/photo-1448375240586-882707db888b?w=500&q=80",
        16: "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=500&q=80",
        17: "https://images.unsplash.com/photo-1629814407887-b95222ef2cb0?w=500&q=80",
        18: "https://images.unsplash.com/photo-1606787366850-de6330128bfc?w=500&q=80",
        19: "https://images.unsplash.com/photo-1608958416715-bc44043b3dfb?w=500&q=80",
        20: "https://images.unsplash.com/photo-1564507592333-c60657eea523?w=500&q=80",
        21: "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=500&q=80",
        22: "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=500&q=80",
        23: "https://images.unsplash.com/photo-1559136555-9303baea8ebd?w=500&q=80",
        24: "https://images.unsplash.com/photo-1534604973900-c43ab4c2e0ab?w=500&q=80",
        25: "https://images.unsplash.com/photo-1582407947304-fd86f028f716?w=500&q=80",
        26: "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=500&q=80",
        27: "https://images.unsplash.com/photo-1549877452-9c387ad6491d?w=500&q=80",
        28: "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=500&q=80",
        29: "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=500&q=80",
        30: "https://images.unsplash.com/photo-1615937657715-bc7b4b7962c1?w=500&q=80",
        31: "https://images.unsplash.com/photo-1447933601403-0c6688de566e?w=500&q=80",
        32: "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=500&q=80",
        33: "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=500&q=80",
        34: "https://images.unsplash.com/photo-1511739001486-6bfe10ce785f?w=500&q=80",
        35: "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=500&q=80",
        36: "https://images.unsplash.com/photo-1552566626-52f8b828add9?w=500&q=80",
        37: "https://images.unsplash.com/photo-1593508512255-86ab42a8e620?w=500&q=80",
        38: "https://images.unsplash.com/photo-1557142046-c704a3adf364?w=500&q=80",
        39: "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=500&q=80",
        40: "https://images.unsplash.com/photo-1465847899084-d164df4dedc6?w=500&q=80",
        41: "https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=500&q=80",
        42: "https://images.unsplash.com/photo-1521017432531-fbd92d768814?w=500&q=80",
        43: "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=500&q=80",
        44: "https://images.unsplash.com/photo-1559742811-82428954133b?w=500&q=80",
        45: "https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=500&q=80",
        46: "https://images.unsplash.com/photo-1560624052-449f5ddf0c31?w=500&q=80",
        47: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=500&q=80",
        48: "https://images.unsplash.com/photo-1543007630-9710e4a00a20?w=500&q=80",
        49: "https://images.unsplash.com/photo-1473968512647-3e447244af8f?w=500&q=80",
        50: "https://images.unsplash.com/photo-1509722747041-616f39b57569?w=500&q=80",
        51: "https://images.unsplash.com/photo-1544816155-12df9643f363?w=500&q=80",
        52: "https://images.unsplash.com/photo-1578683010236-d716f9a3f461?w=500&q=80"
    }
    type_images_fallback = {
        "museum": "https://picsum.photos/id/1018/400/300",
        "monument": "https://picsum.photos/id/1043/400/300",
        "restaurant": "https://picsum.photos/id/1060/400/300",
        "cafe": "https://picsum.photos/id/1025/400/300",
        "entertainment": "https://picsum.photos/id/1015/400/300"
    }

    for place in matched_places:
        if total_spent + place['price'] <= budget and place['type'] not in visited_types:
            if current_time < 12.0:
                time_str = f"{int(current_time)}:00 صباحاً"
            elif current_time == 12.0:
                time_str = "12:00 ظهراً"
            else:
                time_str = f"{int(current_time - 12)}:00 ظهراً"

            place_image = place_images.get(place['id'], type_images_fallback.get(place['type'], "https://picsum.photos/id/20/400/300"))

            program.append({
                "time": time_str,
                "place_name": place['name'],
                "rating": place['rating'],
                "duration": f"{int(place['avg_duration'])} ساعات" if place['avg_duration'] > 1 else "ساعة واحدة",
                "cost": f"{int(place['price'])} جنيه" if place['price'] > 0 else "مجاناً",
                "image_placeholder": place_image,
                "route_guidance": "كيف أصل؟"
            })

            current_time += place['avg_duration'] + 1.0   
            total_spent += place['price']
            visited_types.add(place['type'])

        if current_time >= 17.0 or len(program) >= 3:
            break

    return program, total_spent

import re

def clean_arabic_text(text: str) -> str:
     
    text = re.sub(r"[أإآ]", "ا", text)
     
    text = re.sub(r"ى", "ي", text)
    
    text = re.sub(r"ة", "ه", text)
    
    text = re.sub(r"[\u064B-\u0652]", "", text)
     
    text = re.sub(r"\s+", " ", text).strip()
    return text



def get_smart_recommendation(city: str, preference: str, budget: float):
    city_mask = df_places['city'] == city
    df_filtered = df_places[city_mask].copy()
    
    if df_filtered.empty:
        return None, 0.0, "CITY_NOT_FOUND"

 
    cleaned_preference = clean_arabic_text(preference)

    filtered_embeddings = places_embeddings[city_mask.values]
   
    user_vector = embedding_model.encode([cleaned_preference])
    
    similarities = cosine_similarity(user_vector, filtered_embeddings)[0]
    
    df_filtered['similarity_score'] = similarities
    df_sorted = df_filtered.sort_values(by='similarity_score', ascending=False)
    matched_places = df_sorted.to_dict(orient='records')

    program, total_cost = generate_timeline(matched_places, budget)
    
    if not program:
        return None, 0.0, "BUDGET_TOO_LOW"
        
    return program, total_cost, "SUCCESS"