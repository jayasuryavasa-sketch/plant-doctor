"""India-wide crop reference entries for the educational disease guide.

These entries are deliberately separate from model labels. A guide entry helps
someone learn what to inspect; it never means the local image model can diagnose
that crop or condition.
"""

INDIA_CROP_TOPICS = {
    "Rice": ["Blast", "Bacterial Leaf Blight", "Brown Spot"],
    "Wheat": ["Rust", "Powdery Mildew", "Loose Smut"],
    "Maize": ["Northern Leaf Blight", "Common Rust", "Downy Mildew"],
    "Sorghum": ["Grain Mold", "Anthracnose"],
    "Pearl Millet": ["Downy Mildew", "Ergot"],
    "Finger Millet": ["Blast", "Leaf Spot"],
    "Chickpea": ["Fusarium Wilt", "Ascochyta Blight"],
    "Pigeon Pea": ["Fusarium Wilt", "Sterility Mosaic"],
    "Green Gram": ["Yellow Mosaic", "Powdery Mildew"],
    "Black Gram": ["Yellow Mosaic", "Leaf Crinkle"],
    "Groundnut": ["Early Leaf Spot", "Rust"],
    "Soybean": ["Rust", "Yellow Mosaic"],
    "Mustard": ["Alternaria Blight", "White Rust"],
    "Sesame": ["Phyllody", "Leaf Spot"],
    "Sunflower": ["Alternaria Blight", "Downy Mildew"],
    "Cotton": ["Bacterial Blight", "Leaf Curl Virus"],
    "Sugarcane": ["Red Rot", "Smut"],
    "Jute": ["Stem Rot", "Anthracnose"],
    "Tomato": ["Early Blight", "Late Blight", "Leaf Curl"],
    "Chilli": ["Anthracnose", "Leaf Curl"],
    "Potato": ["Late Blight", "Early Blight"],
    "Brinjal": ["Phomopsis Blight", "Bacterial Wilt"],
    "Onion": ["Purple Blotch", "Downy Mildew"],
    "Okra": ["Yellow Vein Mosaic", "Powdery Mildew"],
    "Cabbage": ["Black Rot", "Downy Mildew"],
    "Banana": ["Sigatoka Leaf Spot", "Fusarium Wilt"],
    "Mango": ["Anthracnose", "Powdery Mildew"],
    "Citrus": ["Citrus Canker", "Greening"],
    "Grapes": ["Downy Mildew", "Powdery Mildew"],
    "Pomegranate": ["Bacterial Blight", "Wilt Complex"],
    "Coconut": ["Bud Rot", "Leaf Rot"],
    "Cashew": ["Tea Mosquito Bug Damage", "Anthracnose"],
    "Turmeric": ["Leaf Blotch", "Rhizome Rot"],
    "Ginger": ["Soft Rot", "Leaf Spot"],
}


def india_crop_guides() -> list[dict]:
    """Create cautious, searchable cards for commonly grown Indian crops."""
    entries: list[dict] = []
    for plant, conditions in INDIA_CROP_TOPICS.items():
        for condition in conditions:
            slug = f"india-{plant}-{condition}".lower().replace(" ", "-")
            entries.append({
                "slug": slug,
                "model_class": "",
                "plant": plant,
                "name": condition,
                "status": "Reference guide",
                "severity": "Needs local review",
                "description": f"A common {condition.lower()} topic for {plant.lower()}. Use this card to compare symptoms, not as a confirmed diagnosis.",
                "symptoms": ["Compare several leaves, including new and older growth", "Look for a repeating pattern and whether it is spreading"],
                "causes": ["The exact cause needs field inspection", "Disease, pests, weather and nutrition can look alike"],
                "spread": "Avoid moving affected plant material and clean tools until the cause is assessed.",
                "conditions": ["Recent rain or humidity", "Insect activity", "Plant stress or poor airflow"],
                "immediate_steps": ["Take clear close-ups and a whole-plant photo", "Note the crop, variety, location and when symptoms started"],
                "treatment": ["Ask the nearest Krishi Vigyan Kendra, agriculture officer or qualified adviser for crop-specific advice", "Use only products registered for that crop and exactly as their labels direct"],
                "organic": ["Keep foliage dry when practical", "Improve sanitation and airflow"],
                "avoid": ["Using a photo result as a pesticide prescription", "Mixing products without qualified advice"],
                "prevention": ["Use healthy planting material", "Monitor regularly and clean tools between plants"],
                "care": ["Water appropriately for the crop", "Keep records of symptoms and actions"],
                "notes": "India-wide reference entry. The free local photo model does not yet diagnose every crop in this guide.",
            })
    return entries
