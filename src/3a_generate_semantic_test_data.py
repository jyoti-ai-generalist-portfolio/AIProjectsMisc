import os

# Create the output directory
output_dir = "data/semantic_data"
os.makedirs(output_dir, exist_ok=True)

# Define 10 categories with 10 variations each to test semantic overlap
dataset = {
    "Space & Astronomy": [
        "The James Webb Space Telescope captures deep field images of early galaxies.",
        "Mars rover Perseverance searches for signs of ancient microbial life in Jezero Crater.",
        "Black holes warp spacetime so severely that not even light can escape their gravity.",
        "The life cycle of a star ends in either a white dwarf, neutron star, or supernova.",
        "Exoplanets orbiting habitable zones might harbor liquid water and life.",
        "NASA plans to establish a permanent human presence on the Moon via the Artemis program.",
        "The rings of Saturn are composed mostly of ice particles and rocky debris.",
        "Light-years measure the distance light travels in one Earth year.",
        "The Milky Way is a barred spiral galaxy containing billions of planetary systems.",
        "Comets leave a glowing trail of gas and dust as they approach the Sun."
    ],
    "Cooking & Culinary Arts": [
        "A perfect sourdough bread requires a mature starter and long fermentation.",
        "Searing a steak at high heat locks in juices and creates a flavorful crust.",
        "Emulsifying olive oil and egg yolks creates a smooth, homemade mayonnaise.",
        "Al dente pasta should be firm to the bite when cooked properly.",
        "Baking requires precise measurements of flour, yeast, and water.",
        "Fermenting cabbage with salt and spices yields traditional Korean kimchi.",
        "A chef's knife should be kept sharp to ensure clean, safe cuts.",
        "Reducing a balsamic glaze creates a thick, sweet drizzle for salads.",
        "Simmering chicken bones for hours extracts collagen for a rich stock.",
        "Blanching vegetables in boiling water then shocking them in ice preserves color."
    ],
    "Personal Finance & Investing": [
        "Diversifying your portfolio across stocks and bonds reduces financial risk.",
        "Compound interest allows your savings to grow exponentially over long periods.",
        "Opening a retirement account early maximizes your long-term capital gains.",
        "Inflation erodes the purchasing power of cash kept under a mattress.",
        "Index funds track a specific market index and offer low fee structures.",
        "Real estate investing can provide passive income through monthly rental properties.",
        "A high-yield savings account offers better returns than traditional checking accounts.",
        "Understanding your debt-to-income ratio is crucial before applying for a mortgage.",
        "Cryptocurrency markets experience high volatility compared to blue-chip stocks.",
        "Budgeting using the 50/30/20 rule helps allocate income efficiently."
    ],
    "Artificial Intelligence & Technology": [
        "Large language models use transformers to predict the next token in text.",
        "Neural networks require massive amounts of labeled data for training.",
        "Semantic search algorithms understand user intent rather than just keywords.",
        "Vector databases index text embeddings to allow fast similarity searches.",
        "Machine learning models can suffer from overfitting if trained too long.",
        "Computer vision enables autonomous vehicles to detect pedestrians and signs.",
        "Natural language processing bridges the gap between human speech and computer logic.",
        "Reinforcement learning trains agents using a system of rewards and penalties.",
        "Generative AI can synthesize realistic images from simple text prompts.",
        "Cloud computing allows developers to scale infrastructure dynamically."
    ],
    "Fitness & Exercise Science": [
        "Progressive overload is essential for building muscle hypertrophy over time.",
        "High-intensity interval training burns calories efficiently in short workouts.",
        "Proper hydration prevents muscle cramps and maintains athletic performance.",
        "Deadlifts target the posterior chain, including hamstrings and glutes.",
        "Cardiovascular exercise strengthens the heart and improves lung capacity.",
        "Yoga enhances flexibility, balance, and core strength through poses.",
        "Rest days are critical for muscle recovery and preventing overtraining injuries.",
        "Tracking your heart rate zones helps optimize aerobic endurance training.",
        "Calisthenics utilizes bodyweight movements like pull-ups and push-ups.",
        "Stretching after a workout helps reduce delayed onset muscle soreness."
    ],
    "Gardening & Plant Care": [
        "Succulents require well-draining soil and infrequent, deep watering cycles.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into plant energy.",
        "Pruning dead branches encourages new growth and improves air circulation.",
        "Nitrogen, phosphorus, and potassium are primary nutrients in plant fertilizer.",
        "Overwatering indoor plants often leads to root rot and yellowing leaves.",
        "Companion planting pairs tomatoes with basil to naturally repel pests.",
        "Monstera plants thrive in bright, indirect sunlight and high humidity.",
        "Composting organic kitchen scraps creates nutrient-rich soil amendments.",
        "Pollinators like bees and butterflies are vital for vegetable garden yields.",
        "Propagating plant cuttings in water allows new root systems to develop."
    ],
    "Ancient History & Archaeology": [
        "The Rosetta Stone allowed scholars to decipher Egyptian hieroglyphs.",
        "The Roman Colosseum hosted gladiatorial combats and public spectacles.",
        "Julius Caesar crossed the Rubicon, triggering a civil war in Rome.",
        "The Code of Hammurabi is one of the earliest written legal systems.",
        "Archaeologists excavated Pompeii to study daily life frozen by volcanic ash.",
        "The Silk Road facilitated trade and cultural exchange across Asia and Europe.",
        "Alexander the Great conquered a vast empire stretching to India.",
        "The Mayan civilization developed advanced mathematics and astronomical calendars.",
        "Cuneiform script was pressed into clay tablets in ancient Mesopotamia.",
        "The Great Pyramid of Giza stood as the tallest man-made structure for millennia."
    ],
    "Environmental Science & Climate": [
        "Melting glaciers contribute significantly to rising global sea levels.",
        "Renewable energy sources include solar, wind, and hydroelectric power.",
        "Deforestation reduces the planet's capacity to absorb carbon dioxide emissions.",
        "The greenhouse effect traps solar radiation within the Earth's atmosphere.",
        "Coral bleaching occurs when rising ocean temperatures stress marine ecosystems.",
        "Microplastics contaminate aquatic food chains and threaten marine life.",
        "Carbon capture technology aims to remove CO2 directly from industrial sources.",
        "Biodiversity loss destabilizes ecosystems and reduces environmental resilience.",
        "Electric vehicles reduce tailpipe emissions compared to internal combustion engines.",
        "Sustainable agriculture focuses on soil conservation and minimizing pesticide use."
    ],
    "Music Theory & Instruments": [
        "A major scale follows a specific pattern of whole and half steps.",
        "The acoustic guitar produces sound through the vibration of its strings.",
        "Syncopation places rhythmic emphasis on the off-beats of a musical measure.",
        "A piano has 88 keys spanning more than seven octaves.",
        "Musicians use a metronome to maintain a steady tempo during practice.",
        "Chords are built by stacking thirds on top of a root note.",
        "The violin is the smallest and highest-pitched member of the string family.",
        "Jazz music relies heavily on improvisation and complex chord extensions.",
        "A crescendo indicates a gradual increase in musical volume.",
        "Sheet music uses clefs to dictate the pitch of written notes."
    ],
    "Medical Science & Health": [
        "Antibiotics kill bacteria but are completely ineffective against viral infections.",
        "The human heart beats roughly 100,000 times a day to pump blood.",
        "White blood cells are the primary defense mechanism of the immune system.",
        "Vaccines train the immune system to recognize and combat specific pathogens.",
        "Chronic stress elevates cortisol levels, negatively impacting overall health.",
        "Insulin regulates blood glucose levels by helping cells absorb sugar.",
        "The nervous system transmits electrical signals between the brain and body.",
        "Getting eight hours of sleep support cognitive function and tissue repair.",
        "Hypertension increases the long-term risk of stroke and cardiovascular disease.",
        "Vitamin D is synthesized by the skin upon exposure to ultraviolet sunlight."
    ]
}

# Write the 100 text files
file_count = 0
for category, sentences in dataset.items():
    # Clean category name for file naming
    cat_clean = category.lower().replace(" & ", "_").replace(" ", "_")
    for idx, sentence in enumerate(sentences, start=1):
        filename = f"{cat_clean}_{idx:02d}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sentence)
        file_count += 1

print(f"Successfully generated {file_count} text files inside the '{output_dir}' directory!")
