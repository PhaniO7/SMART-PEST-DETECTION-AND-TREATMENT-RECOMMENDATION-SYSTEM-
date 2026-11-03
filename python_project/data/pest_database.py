"""
================================================================================
FILE: data/pest_database.py
PURPOSE:
    Contains the agricultural pest & treatment knowledge base for the TOP 20
    IP102 insect pest classes across all 8 IP102 host crops:
        1. Rice    2. Corn (Maize)   3. Wheat    4. Beet
        5. Alfalfa 6. Vitis (Grapes) 7. Citrus   8. Mango

    IMPORTANT ACADEMIC NOTE FOR STUDENT VERIFICATION:
    The official IP102 dataset (CVPR 2019) contains only images, class labels,
    and Pascal VOC bounding boxes — it does NOT include pesticide or agronomic
    treatment data. The treatment records below (Organic, Biological, and
    Chemical options, efficacy %, Pre-Harvest Interval [PHI] days, dosage per
    acre, and approximate cost in INR) have been compiled from standard Indian
    agricultural references.
    >>> PLEASE VERIFY AND CROSS-CHECK THESE DOSAGES, PHI DAYS, AND COSTS WITH
    >>> OFFICIAL ICAR (Indian Council of Agricultural Research), TNAU Agritech
    >>> Portal, OR YOUR STATE AGRICULTURAL UNIVERSITY (SAU) PACKAGE OF PRACTICES
    >>> BEFORE FIELD USE OR FINAL THESIS SUBMISSION.

CONCEPTS USED:
    - Structured Knowledge Base Records (# [DATABASE], # [DATASET], # [TREATMENT])
    - Populating our Custom 31-Bucket Hash Table (# [HASH_TABLE])

HOW IT CONNECTS TO OTHER FILES:
    - Imported by `dsa_pipeline.py`, `inference.py`, and `app.py` to populate
      `CustomPestHashTable` and retrieve pest details in O(1) average time.
    - Imported by `prepare_dataset.py` to map IP102 class IDs to host crops.
================================================================================
"""

# Import typing helpers for clear structure definitions
from typing import Any, Dict, List

# Import our custom 31-bucket Hash Table from dsa_pipeline.py
from dsa_pipeline import CustomPestHashTable

# ==============================================================================
# MAP OF ALL 102 IP102 CLASS IDs (1-BASED) TO THEIR OFFICIAL HOST CROP
# Per the official IP102 paper (Table 1 in Wu et al., CVPR 2019):
#   Classes 1..14   -> Rice
#   Classes 15..27  -> Corn
#   Classes 28..36  -> Wheat
#   Classes 37..44  -> Beet
#   Classes 45..57  -> Alfalfa
#   Classes 58..73  -> Vitis (Grapes)
#   Classes 74..92  -> Citrus
#   Classes 93..102 -> Mango
# ==============================================================================

# # [DATASET] Helper function mapping any 1-based IP102 class ID (1..102) to its crop
def get_crop_for_ip102_class_id(ip102_id_1based: int) -> str:
    """Returns the official IP102 host crop name for a 1-based IP102 class ID (1..102)."""
    if 1 <= ip102_id_1based <= 14:
        return "Rice"
    if 15 <= ip102_id_1based <= 27:
        return "Corn"
    if 28 <= ip102_id_1based <= 36:
        return "Wheat"
    if 37 <= ip102_id_1based <= 44:
        return "Beet"
    if 45 <= ip102_id_1based <= 57:
        return "Alfalfa"
    if 58 <= ip102_id_1based <= 73:
        return "Vitis"
    if 74 <= ip102_id_1based <= 92:
        return "Citrus"
    return "Mango"


# ==============================================================================
# TOP 20 IP102 PEST CLASSES DATABASE (ACROSS ALL 8 CROPS)
# # [DATABASE] # [TREATMENT] # [CHANGE_HERE] Verify dosages/costs with ICAR/TNAU guides
# ==============================================================================

IP102_TOP20_PEST_DATABASE: List[Dict[str, Any]] = [
    # --------------------------------------------------------------------------
    # CLASS 0 (IP102 ID #1) — RICE: Rice Leaf Roller
    # --------------------------------------------------------------------------
    {
        "class_index": 0,
        "ip102_id": 1,
        "crop": "Rice",
        "common_name": "rice leaf roller",
        "display_name": "Rice Leaf Roller",
        "scientific_name": "Cnaphalocrocis medinalis",
        "life_stage": "Larva (Caterpillar) & Adult Moth",
        "symptoms": (
            "Larvae fold rice leaves longitudinally with silk threads and scrape the green "
            "chlorophyll layer from inside, leaving papery white linear streaks on the blade."
        ),
        "treatments": [
            {
                "name": "Azadirachtin 1% EC (Neem Botanical Spray)",
                "active_ingredient": "Azadirachtin 10,000 ppm",
                "category": "Organic",
                "efficacy_percent": 79.0,
                "phi_days": 3,
                "cost_per_acre_inr": 420,
                "dosage": "400 ml in 200 L water per acre",
                "safety_precautions": "Spray in early morning or evening; safe for natural spider predators.",
            },
            {
                "name": "Trichogramma chilonis Egg Parasitoid + Bt Kurstaki",
                "active_ingredient": "Bacillus thuringiensis var. kurstaki 1.5% WP",
                "category": "Biological",
                "efficacy_percent": 86.0,
                "phi_days": 1,
                "cost_per_acre_inr": 540,
                "dosage": "500 g Bt WP/acre + 50,000 Trichogramma cards/acre",
                "safety_precautions": "Do not mix biological spray with chemical fungicides.",
            },
            {
                "name": "Chlorantraniliprole 18.5% SC (Coragen)",
                "active_ingredient": "Chlorantraniliprole 18.5% SC",
                "category": "Chemical",
                "efficacy_percent": 95.0,
                "phi_days": 14,
                "cost_per_acre_inr": 880,
                "dosage": "60 ml in 200 L water per acre",
                "safety_precautions": "Wear gloves and mask; avoid draining paddy water into fish ponds for 7 days.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 1 (IP102 ID #5) — RICE: Brown Planthopper
    # --------------------------------------------------------------------------
    {
        "class_index": 1,
        "ip102_id": 5,
        "crop": "Rice",
        "common_name": "brown plant hopper",
        "display_name": "Rice Brown Planthopper (BPH)",
        "scientific_name": "Nilaparvata lugens",
        "life_stage": "Nymph & Adult",
        "symptoms": (
            "Congregates at the base of rice tillers above water level, sucking phloem sap and "
            "causing circular yellowing that rapidly dries into brownish 'hopperburn' patches."
        ),
        "treatments": [
            {
                "name": "Neem Seed Kernel Extract (NSKE 5%) + Alleyway Parting",
                "active_ingredient": "Azadirachtin 3000 ppm",
                "category": "Organic",
                "efficacy_percent": 76.0,
                "phi_days": 3,
                "cost_per_acre_inr": 390,
                "dosage": "600 ml in 200 L water directed at tiller base",
                "safety_precautions": "Drain standing water for 3 days to expose tiller bases to sunlight.",
            },
            {
                "name": "Metarhizium anisopliae 1.15% WP Entomopathogenic Fungus",
                "active_ingredient": "Metarhizium anisopliae (1x10^8 CFU/g)",
                "category": "Biological",
                "efficacy_percent": 84.0,
                "phi_days": 2,
                "cost_per_acre_inr": 480,
                "dosage": "1.0 kg in 200 L water per acre",
                "safety_precautions": "Apply during humid evening hours for fungal spore germination.",
            },
            {
                "name": "Pymetrozine 50% WG (Chess)",
                "active_ingredient": "Pymetrozine 50% WG",
                "category": "Chemical",
                "efficacy_percent": 96.0,
                "phi_days": 15,
                "cost_per_acre_inr": 790,
                "dosage": "120 g in 200 L water per acre",
                "safety_precautions": "Direct spray nozzle toward lower stem base; wear protective apron.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 2 (IP102 ID #8) — RICE: Asiatic Rice Borer (Yellow Stem Borer)
    # --------------------------------------------------------------------------
    {
        "class_index": 2,
        "ip102_id": 8,
        "crop": "Rice",
        "common_name": "asiatic rice borer",
        "display_name": "Asiatic Rice Stem Borer",
        "scientific_name": "Chilo suppressalis",
        "life_stage": "Larva (Stem Borer) & Adult",
        "symptoms": (
            "Larvae bore into the central rice stem, severing the growing shoot to cause dried "
            "'deadhearts' at vegetative stage and chaffy 'whiteheads' at panicle emergence."
        ),
        "treatments": [
            {
                "name": "Pheromone Funnel Traps + Neem Oil 1500 ppm",
                "active_ingredient": "Z-11-Hexadecenal Lure + Azadirachtin",
                "category": "Organic",
                "efficacy_percent": 77.0,
                "phi_days": 2,
                "cost_per_acre_inr": 460,
                "dosage": "8 traps/acre + 500 ml Neem Oil/acre",
                "safety_precautions": "Clip tip of rice seedlings before transplanting to remove egg masses.",
            },
            {
                "name": "Beauveria bassiana 1.15% WP Bio-Larvicide",
                "active_ingredient": "Beauveria bassiana spores",
                "category": "Biological",
                "efficacy_percent": 83.0,
                "phi_days": 2,
                "cost_per_acre_inr": 510,
                "dosage": "1.0 kg in 200 L water per acre",
                "safety_precautions": "Safe for honeybees and aquatic frogs; store bio-powder in cool shade.",
            },
            {
                "name": "Cartap Hydrochloride 4% GR / Flubendiamide 20% WG",
                "active_ingredient": "Flubendiamide 20% WG",
                "category": "Chemical",
                "efficacy_percent": 94.0,
                "phi_days": 20,
                "cost_per_acre_inr": 860,
                "dosage": "50 g in 200 L water per acre",
                "safety_precautions": "Do not apply within 20 days of grain harvest; wear face shield.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 3 (IP102 ID #15) — CORN: Grub (White Grub)
    # --------------------------------------------------------------------------
    {
        "class_index": 3,
        "ip102_id": 15,
        "crop": "Corn",
        "common_name": "grub",
        "display_name": "Corn White Grub",
        "scientific_name": "Holotrichia oblita",
        "life_stage": "Soil Larva (C-Shaped Grub) & Adult Beetle",
        "symptoms": (
            "C-shaped white larvae feed on maize fibrous roots underground while adults chew "
            "irregular notches along young corn leaf margins, causing wilting and lodging."
        ),
        "treatments": [
            {
                "name": "Neem Cake Soil Amendment + Light Trap Collection",
                "active_ingredient": "De-oiled Azadirachta indica Cake",
                "category": "Organic",
                "efficacy_percent": 78.0,
                "phi_days": 0,
                "cost_per_acre_inr": 650,
                "dosage": "100 kg Neem Cake incorporated into furrow per acre",
                "safety_precautions": "Deep summer ploughing exposes grubs to predatory birds.",
            },
            {
                "name": "Heterorhabditis indica Entomopathogenic Nematodes (EPN)",
                "active_ingredient": "Heterorhabditis indica infective juveniles",
                "category": "Biological",
                "efficacy_percent": 87.0,
                "phi_days": 1,
                "cost_per_acre_inr": 620,
                "dosage": "2.0 kg EPN formulation drenched in root zone per acre",
                "safety_precautions": "Apply in moist soil during evening; completely non-toxic to humans.",
            },
            {
                "name": "Thiamethoxam 30% FS / Chlorantraniliprole Drench",
                "active_ingredient": "Chlorantraniliprole 18.5% SC",
                "category": "Chemical",
                "efficacy_percent": 93.0,
                "phi_days": 18,
                "cost_per_acre_inr": 920,
                "dosage": "80 ml in 200 L water drenched at root collar per acre",
                "safety_precautions": "Wear rubber gloves during drenching; wash hands thoroughly after use.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 4 (IP102 ID #18) — CORN: Corn Borer (Asian Corn Borer)
    # --------------------------------------------------------------------------
    {
        "class_index": 4,
        "ip102_id": 18,
        "crop": "Corn",
        "common_name": "corn borer",
        "display_name": "Asian Corn Borer",
        "scientific_name": "Ostrinia furnacalis",
        "life_stage": "Larva (Caterpillar) & Adult",
        "symptoms": (
            "Produces characteristic 'shot-hole' rows across unrolling maize whorl leaves "
            "and bores into stalks and tassels, leaving sawdust-like frass at entry holes."
        ),
        "treatments": [
            {
                "name": "Azadirachtin 3000 ppm Whorl Directed Spray",
                "active_ingredient": "Azadirachtin 0.3% EC",
                "category": "Organic",
                "efficacy_percent": 80.0,
                "phi_days": 3,
                "cost_per_acre_inr": 440,
                "dosage": "500 ml in 200 L water directed into leaf whorl",
                "safety_precautions": "Apply at early whorl stage before larvae tunnel deep into stalks.",
            },
            {
                "name": "Trichogramma ostriniae Cards + Bt Kurstaki",
                "active_ingredient": "Bacillus thuringiensis + Trichogramma",
                "category": "Biological",
                "efficacy_percent": 88.0,
                "phi_days": 1,
                "cost_per_acre_inr": 530,
                "dosage": "400 g Bt WP/acre + 2 parasitoid releases",
                "safety_precautions": "Attach Trichogramma cards on underside of middle corn leaves.",
            },
            {
                "name": "Spinetoram 11.7% SC (Delegate) / Emamectin Benzoate 5% SG",
                "active_ingredient": "Emamectin Benzoate 5% SG",
                "category": "Chemical",
                "efficacy_percent": 96.0,
                "phi_days": 10,
                "cost_per_acre_inr": 560,
                "dosage": "80 g Emamectin Benzoate in 200 L water per acre",
                "safety_precautions": "Avoid spraying against wind; wear full-sleeve clothing and eye protection.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 5 (IP102 ID #19) — CORN: Armyworm (Mythimna separata)
    # --------------------------------------------------------------------------
    {
        "class_index": 5,
        "ip102_id": 19,
        "crop": "Corn",
        "common_name": "armyworm",
        "display_name": "Oriental Corn Armyworm",
        "scientific_name": "Mythimna separata",
        "life_stage": "Gregarious Larva (Caterpillar)",
        "symptoms": (
            "Voracious caterpillars skeletonize maize leaves from the margins inward, leaving "
            "only the tough midrib intact and depositing coarse greenish-brown pellets in the whorl."
        ),
        "treatments": [
            {
                "name": "Neem Oil 10,000 ppm + Sand-Lime Whorl Application",
                "active_ingredient": "Azadirachtin 1% EC",
                "category": "Organic",
                "efficacy_percent": 79.0,
                "phi_days": 3,
                "cost_per_acre_inr": 410,
                "dosage": "400 ml in 200 L water per acre",
                "safety_precautions": "Spray in late evening when nocturnal armyworm larvae emerge to feed.",
            },
            {
                "name": "Nomuraea rileyi / Metarhizium anisopliae Bio-Spray",
                "active_ingredient": "Metarhizium rileyi (1x10^8 spores/g)",
                "category": "Biological",
                "efficacy_percent": 87.0,
                "phi_days": 1,
                "cost_per_acre_inr": 490,
                "dosage": "800 g in 200 L water per acre",
                "safety_precautions": "Zero waiting period; safe for beneficial ladybird beetles.",
            },
            {
                "name": "Spinetoram 11.7% SC",
                "active_ingredient": "Spinetoram 11.7% SC",
                "category": "Chemical",
                "efficacy_percent": 97.0,
                "phi_days": 7,
                "cost_per_acre_inr": 940,
                "dosage": "100 ml in 200 L water per acre",
                "safety_precautions": "Toxic to bees during active foraging; spray only after 4 PM.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 6 (IP102 ID #24) — CORN: Corn Leaf Aphid
    # --------------------------------------------------------------------------
    {
        "class_index": 6,
        "ip102_id": 24,
        "crop": "Corn",
        "common_name": "aphids",
        "display_name": "Corn Leaf & Tassel Aphid",
        "scientific_name": "Rhopalosiphum maidis",
        "life_stage": "Nymph & Winged/Wingless Adult",
        "symptoms": (
            "Dense bluish-green colonies cluster inside the upper whorl and tassel, sucking "
            "sap and excreting sticky honeydew that coats leaves with black sooty mold."
        ),
        "treatments": [
            {
                "name": "Potassium Salts of Fatty Acids (Insecticidal Soap) + Neem",
                "active_ingredient": "Potassium Oleate 2% + Azadirachtin",
                "category": "Organic",
                "efficacy_percent": 83.0,
                "phi_days": 1,
                "cost_per_acre_inr": 380,
                "dosage": "500 ml in 200 L water per acre",
                "safety_precautions": "Ensure thorough coverage of leaf undersides and tassel base.",
            },
            {
                "name": "Lecanicillium lecanii (Verticillium) 1.15% WP",
                "active_ingredient": "Lecanicillium lecanii (1x10^8 CFU/g)",
                "category": "Biological",
                "efficacy_percent": 86.0,
                "phi_days": 1,
                "cost_per_acre_inr": 450,
                "dosage": "1.0 kg in 200 L water per acre",
                "safety_precautions": "Conserves natural Chrysoperla (green lacewing) and Coccinellid predators.",
            },
            {
                "name": "Flonicamid 50% WG (Ulala)",
                "active_ingredient": "Flonicamid 50% WG",
                "category": "Chemical",
                "efficacy_percent": 95.0,
                "phi_days": 10,
                "cost_per_acre_inr": 640,
                "dosage": "60 g in 200 L water per acre",
                "safety_precautions": "Selective anti-feeding action; wear gloves and avoid skin contact.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 7 (IP102 ID #30) — WHEAT: English Grain Aphid
    # --------------------------------------------------------------------------
    {
        "class_index": 7,
        "ip102_id": 30,
        "crop": "Wheat",
        "common_name": "english grain aphid",
        "display_name": "Wheat English Grain Aphid",
        "scientific_name": "Sitobion avenae",
        "life_stage": "Nymph & Adult",
        "symptoms": (
            "Colonies feed on wheat flag leaves and developing grain spikes, causing yellow "
            "stippling, flag-leaf curling, shriveled grains, and transmission of Barley Yellow Dwarf Virus."
        ),
        "treatments": [
            {
                "name": "Azadirachtin 1500 ppm Foliar Spray + Yellow Sticky Traps",
                "active_ingredient": "Azadirachtin 0.15% EC",
                "category": "Organic",
                "efficacy_percent": 81.0,
                "phi_days": 2,
                "cost_per_acre_inr": 360,
                "dosage": "15 sticky traps/acre + 500 ml Neem EC/acre",
                "safety_precautions": "Install yellow sticky traps 15 cm above the wheat canopy.",
            },
            {
                "name": "Lecanicillium lecanii Entomopathogenic Bio-Control",
                "active_ingredient": "Lecanicillium lecanii 1.15% WP",
                "category": "Biological",
                "efficacy_percent": 85.0,
                "phi_days": 1,
                "cost_per_acre_inr": 440,
                "dosage": "800 g in 200 L water per acre",
                "safety_precautions": "Apply when aphid count exceeds 10-15 per earhead.",
            },
            {
                "name": "Thiamethoxam 25% WG (Actara)",
                "active_ingredient": "Thiamethoxam 25% WG",
                "category": "Chemical",
                "efficacy_percent": 95.0,
                "phi_days": 14,
                "cost_per_acre_inr": 340,
                "dosage": "40 g in 200 L water per acre",
                "safety_precautions": "Do not spray during active bee visitation; observe 14-day pre-harvest interval.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 8 (IP102 ID #34) — WHEAT: Red Spider (Wheat Spider Mite)
    # --------------------------------------------------------------------------
    {
        "class_index": 8,
        "ip102_id": 34,
        "crop": "Wheat",
        "common_name": "red spider",
        "display_name": "Wheat Red Spider Mite",
        "scientific_name": "Petrobia latens",
        "life_stage": "Larva, Nymph & Adult Mite",
        "symptoms": (
            "Tiny reddish-brown mites pierce leaf cells, producing fine silvery-yellow speckles, "
            "bronze discoloration of wheat blades, and fine webbing under dry weather."
        ),
        "treatments": [
            {
                "name": "Wettable Sulphur 80% WP (Organic Approved Miticide)",
                "active_ingredient": "Micronized Elemental Sulphur 80% WP",
                "category": "Organic",
                "efficacy_percent": 84.0,
                "phi_days": 3,
                "cost_per_acre_inr": 320,
                "dosage": "800 g in 200 L water per acre",
                "safety_precautions": "Do not apply if ambient temperature exceeds 35°C to avoid leaf scorching.",
            },
            {
                "name": "Hirsutella thompsonii / Beauveria Bio-Acaricide",
                "active_ingredient": "Hirsutella thompsonii 1.0% WP",
                "category": "Biological",
                "efficacy_percent": 82.0,
                "phi_days": 1,
                "cost_per_acre_inr": 480,
                "dosage": "1.0 kg in 200 L water per acre",
                "safety_precautions": "Safe for predatory phytoseiid mites.",
            },
            {
                "name": "Spiromesifen 22.9% SC (Oberon) / Propargite 57% EC",
                "active_ingredient": "Spiromesifen 22.9% SC",
                "category": "Chemical",
                "efficacy_percent": 95.0,
                "phi_days": 14,
                "cost_per_acre_inr": 780,
                "dosage": "150 ml in 200 L water per acre",
                "safety_precautions": "Coat both upper and lower leaf surfaces; wear protective goggles.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 9 (IP102 ID #38) — BEET: Black Cutworm (Agrotis ipsilon)
    # --------------------------------------------------------------------------
    {
        "class_index": 9,
        "ip102_id": 38,
        "crop": "Beet",
        "common_name": "black cutworm",
        "display_name": "Beet Black Cutworm",
        "scientific_name": "Agrotis ipsilon",
        "life_stage": "Nocturnal Larva (Greasy Cutworm)",
        "symptoms": (
            "Greasy dark-grey caterpillars hide in soil cracks by day and sever young beet "
            "stems at the soil line at night, or chew ragged holes in lower leaves."
        ),
        "treatments": [
            {
                "name": "Fermented Jaggery + Neem Bran Poison Bait (Organic)",
                "active_ingredient": "Wheat Bran + Jaggery + Azadirachtin",
                "category": "Organic",
                "efficacy_percent": 80.0,
                "phi_days": 1,
                "cost_per_acre_inr": 410,
                "dosage": "10 kg moistened bait broadcast along rows at dusk per acre",
                "safety_precautions": "Broadcast bait in late evening so it remains moist for night-feeding larvae.",
            },
            {
                "name": "Steinernema carpocapsae Entomopathogenic Nematodes",
                "active_ingredient": "Steinernema carpocapsae",
                "category": "Biological",
                "efficacy_percent": 86.0,
                "phi_days": 1,
                "cost_per_acre_inr": 590,
                "dosage": "1.5 kg in 200 L water drenched along plant collars",
                "safety_precautions": "Irrigate lightly before and after application.",
            },
            {
                "name": "Chlorantraniliprole 18.5% SC Collar Drench",
                "active_ingredient": "Chlorantraniliprole 18.5% SC",
                "category": "Chemical",
                "efficacy_percent": 94.0,
                "phi_days": 14,
                "cost_per_acre_inr": 850,
                "dosage": "60 ml in 200 L water directed at plant base",
                "safety_precautions": "Avoid livestock grazing on treated beet tops for 14 days.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 10 (IP102 ID #40) — BEET: Beet Armyworm (Spodoptera exigua)
    # --------------------------------------------------------------------------
    {
        "class_index": 10,
        "ip102_id": 40,
        "crop": "Beet",
        "common_name": "beet army worm",
        "display_name": "Beet Armyworm",
        "scientific_name": "Spodoptera exigua",
        "life_stage": "Larva (Caterpillar) & Adult",
        "symptoms": (
            "Young larvae feed gregariously under fine webbing, skeletonizing beet leaves "
            "into transparent 'windowpanes' before older larvae chew large irregular holes."
        ),
        "treatments": [
            {
                "name": "Azadirachtin 10,000 ppm + Spodoptera Pheromone Traps",
                "active_ingredient": "Azadirachtin 1% EC",
                "category": "Organic",
                "efficacy_percent": 81.0,
                "phi_days": 3,
                "cost_per_acre_inr": 490,
                "dosage": "400 ml/acre + 6 pheromone traps/acre",
                "safety_precautions": "Hand-pick cottony egg masses on leaf undersides during scouting.",
            },
            {
                "name": "Spodoptera exigua Nucleopolyhedrovirus (SeNPV) + Bt",
                "active_ingredient": "SeNPV 1x10^9 POB/ml",
                "category": "Biological",
                "efficacy_percent": 89.0,
                "phi_days": 1,
                "cost_per_acre_inr": 560,
                "dosage": "250 LE (Larval Equivalent) in 200 L water per acre",
                "safety_precautions": "Spray in evening to prevent UV degradation of viral occlusion bodies.",
            },
            {
                "name": "Indoxacarb 14.5% SC / Emamectin Benzoate 5% SG",
                "active_ingredient": "Indoxacarb 14.5% SC",
                "category": "Chemical",
                "efficacy_percent": 95.0,
                "phi_days": 10,
                "cost_per_acre_inr": 680,
                "dosage": "120 ml in 200 L water per acre",
                "safety_precautions": "Rotate mode of action to prevent Spodoptera resistance.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 11 (IP102 ID #49) — ALFALFA: Alfalfa Weevil (Hypera postica)
    # --------------------------------------------------------------------------
    {
        "class_index": 11,
        "ip102_id": 49,
        "crop": "Alfalfa",
        "common_name": "alfalfa weevil",
        "display_name": "Alfalfa Leaf Weevil",
        "scientific_name": "Hypera postica",
        "life_stage": "Green Larva & Snout Beetle Adult",
        "symptoms": (
            "Green larvae with a white dorsal stripe chew pinholes in terminal leaf buds and "
            "skeletonize upper alfalfa leaflets, giving heavily infested fields a frosted grey look."
        ),
        "treatments": [
            {
                "name": "Early Harvest Cutting + Neem Oil 3000 ppm Spray",
                "active_ingredient": "Azadirachtin 0.3% EC",
                "category": "Organic",
                "efficacy_percent": 79.0,
                "phi_days": 2,
                "cost_per_acre_inr": 390,
                "dosage": "500 ml in 200 L water on regrowth per acre",
                "safety_precautions": "Early cutting starves young larvae and exposes them to sunlight.",
            },
            {
                "name": "Beauveria bassiana 1.15% WP + Bathyplectes Conservation",
                "active_ingredient": "Beauveria bassiana spores",
                "category": "Biological",
                "efficacy_percent": 85.0,
                "phi_days": 1,
                "cost_per_acre_inr": 470,
                "dosage": "1.0 kg in 200 L water per acre",
                "safety_precautions": "Safe for dairy cattle forage after 24 hours.",
            },
            {
                "name": "Indoxacarb 14.5% SC (Steward)",
                "active_ingredient": "Indoxacarb 14.5% SC",
                "category": "Chemical",
                "efficacy_percent": 94.0,
                "phi_days": 7,
                "cost_per_acre_inr": 660,
                "dosage": "100 ml in 200 L water per acre",
                "safety_precautions": "Wait full 7-day PHI before harvesting alfalfa for livestock feed.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 12 (IP102 ID #52) — ALFALFA: Tarnished Plant Bug (Lygus lineolaris)
    # --------------------------------------------------------------------------
    {
        "class_index": 12,
        "ip102_id": 52,
        "crop": "Alfalfa",
        "common_name": "tarnished plant bug",
        "display_name": "Alfalfa Tarnished Plant Bug (Lygus)",
        "scientific_name": "Lygus lineolaris",
        "life_stage": "Nymph & Adult Mirid Bug",
        "symptoms": (
            "Adults and nymphs inject toxic saliva into alfalfa buds, flowers, and young pods, "
            "causing bud blasting, flower drop, crinkled leaves, and shriveled seeds."
        ),
        "treatments": [
            {
                "name": "Karanja (Pongamia) + Neem Botanical Emulsion",
                "active_ingredient": "Karanjin + Azadirachtin",
                "category": "Organic",
                "efficacy_percent": 78.0,
                "phi_days": 2,
                "cost_per_acre_inr": 430,
                "dosage": "500 ml in 200 L water per acre",
                "safety_precautions": "Leave uncut border strips as trap crops for predatory damsel bugs.",
            },
            {
                "name": "Beauveria bassiana Myco-Insecticide",
                "active_ingredient": "Beauveria bassiana 1x10^8 CFU/g",
                "category": "Biological",
                "efficacy_percent": 84.0,
                "phi_days": 1,
                "cost_per_acre_inr": 490,
                "dosage": "1.0 kg in 200 L water per acre",
                "safety_precautions": "Spray in evening to protect alkali bees and leafcutter pollinators.",
            },
            {
                "name": "Flonicamid 50% WG",
                "active_ingredient": "Flonicamid 50% WG",
                "category": "Chemical",
                "efficacy_percent": 93.0,
                "phi_days": 10,
                "cost_per_acre_inr": 650,
                "dosage": "60 g in 200 L water per acre",
                "safety_precautions": "Do not apply during peak bloom when bees are actively foraging.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 13 (IP102 ID #57) — ALFALFA: Blister Beetle
    # --------------------------------------------------------------------------
    {
        "class_index": 13,
        "ip102_id": 57,
        "crop": "Alfalfa",
        "common_name": "blister beetle",
        "display_name": "Alfalfa Blister Beetle",
        "scientific_name": "Epicauta gorhami / Mylabris pustulata",
        "life_stage": "Gregarious Adult Beetle",
        "symptoms": (
            "Swarms of elongated black-and-orange beetles devour alfalfa flowers and tender "
            "leaves; beetles contain cantharidin toxin which is hazardous if baled into hay."
        ),
        "treatments": [
            {
                "name": "Neem + Pyrethrum Natural Botanical Deterrent",
                "active_ingredient": "Natural Pyrethrins 0.2% + Azadirachtin",
                "category": "Organic",
                "efficacy_percent": 82.0,
                "phi_days": 2,
                "cost_per_acre_inr": 520,
                "dosage": "400 ml in 200 L water per acre",
                "safety_precautions": "Harvest before flowering to prevent beetle swarms in hay.",
            },
            {
                "name": "Beauveria bassiana + Manual Net Sweeping",
                "active_ingredient": "Beauveria bassiana 1.15% WP",
                "category": "Biological",
                "efficacy_percent": 83.0,
                "phi_days": 1,
                "cost_per_acre_inr": 460,
                "dosage": "1.0 kg in 200 L water per acre",
                "safety_precautions": "Wear gloves when handling beetles as haemolymph causes skin blisters.",
            },
            {
                "name": "Lambda-Cyhalothrin 5% EC / Carbaryl Spot Spray",
                "active_ingredient": "Lambda-Cyhalothrin 5% EC",
                "category": "Chemical",
                "efficacy_percent": 94.0,
                "phi_days": 14,
                "cost_per_acre_inr": 420,
                "dosage": "200 ml in 200 L water per acre on localized swarms",
                "safety_precautions": "Wait for dead beetles to drop off plants before cutting hay.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 14 (IP102 ID #67) — VITIS (GRAPES): Cicada / Grape Leafhopper
    # --------------------------------------------------------------------------
    {
        "class_index": 14,
        "ip102_id": 67,
        "crop": "Vitis",
        "common_name": "cicadas",
        "display_name": "Vitis Grapevine Leafhopper & Cicada",
        "scientific_name": "Erythroneura apicalis / Cryptotympana atrata",
        "life_stage": "Nymph & Adult",
        "symptoms": (
            "Sucks sap from grapevine leaf undersides causing dense white/yellow stippling, "
            "marginal leaf browning, and oviposition slits on young grape canes."
        ),
        "treatments": [
            {
                "name": "Kaolin Clay Particle Film + Neem Oil 3000 ppm",
                "active_ingredient": "Calcined Kaolin 95% WP + Azadirachtin",
                "category": "Organic",
                "efficacy_percent": 81.0,
                "phi_days": 1,
                "cost_per_acre_inr": 580,
                "dosage": "2.5 kg Kaolin + 400 ml Neem in 200 L water per acre",
                "safety_precautions": "Creates a white protective mineral film that deters feeding and egg-laying.",
            },
            {
                "name": "Lecanicillium lecanii + Anagrus Parasitoid Conservation",
                "active_ingredient": "Lecanicillium lecanii 1.15% WP",
                "category": "Biological",
                "efficacy_percent": 85.0,
                "phi_days": 1,
                "cost_per_acre_inr": 510,
                "dosage": "1.0 kg in 200 L water per acre",
                "safety_precautions": "Safe for table grapes close to harvest.",
            },
            {
                "name": "Buprofezin 25% SC (Applaud)",
                "active_ingredient": "Buprofezin 25% SC",
                "category": "Chemical",
                "efficacy_percent": 94.0,
                "phi_days": 15,
                "cost_per_acre_inr": 720,
                "dosage": "300 ml in 200 L water per acre",
                "safety_precautions": "Insect growth regulator; observe 15-day PHI for export grapes.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 15 (IP102 ID #70) — VITIS (GRAPES): Spotted Lanternfly (Lycorma delicatula)
    # --------------------------------------------------------------------------
    {
        "class_index": 15,
        "ip102_id": 70,
        "crop": "Vitis",
        "common_name": "lycorma delicatula",
        "display_name": "Grapevine Spotted Lanternfly",
        "scientific_name": "Lycorma delicatula",
        "life_stage": "Nymph & Adult Planthopper",
        "symptoms": (
            "Large spotted planthoppers pierce grapevine shoots and trunks in groups, causing "
            "weeping sap wounds, wilting canes, heavy honeydew deposits, and black sooty mold."
        ),
        "treatments": [
            {
                "name": "Trunk Circle Traps + Rosemary/Neem Essential Oil Spray",
                "active_ingredient": "Azadirachtin 1% EC + Mechanical Circle Trap",
                "category": "Organic",
                "efficacy_percent": 79.0,
                "phi_days": 2,
                "cost_per_acre_inr": 540,
                "dosage": "500 ml Neem EC in 200 L water per acre + trunk traps",
                "safety_precautions": "Scrape putty-like grey egg masses off vine posts during winter pruning.",
            },
            {
                "name": "Beauveria bassiana Strain GHA Bio-Spray",
                "active_ingredient": "Beauveria bassiana 1.15% WP",
                "category": "Biological",
                "efficacy_percent": 86.0,
                "phi_days": 1,
                "cost_per_acre_inr": 580,
                "dosage": "1.2 kg in 200 L water directed at vine trunks and canopy",
                "safety_precautions": "Most effective against red/black nymphal instars.",
            },
            {
                "name": "Dinotefuran 20% SG / Bifenthrin Directed Spray",
                "active_ingredient": "Dinotefuran 20% SG",
                "category": "Chemical",
                "efficacy_percent": 96.0,
                "phi_days": 14,
                "cost_per_acre_inr": 840,
                "dosage": "100 g in 200 L water per acre",
                "safety_precautions": "Apply only post-bloom; wear respirator and chemical-resistant gloves.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 16 (IP102 ID #75) — CITRUS: Citrus Red Mite (Panonychus citri)
    # --------------------------------------------------------------------------
    {
        "class_index": 16,
        "ip102_id": 75,
        "crop": "Citrus",
        "common_name": "panonchus citri mcgregor",
        "display_name": "Citrus Red Mite",
        "scientific_name": "Panonychus citri",
        "life_stage": "Egg, Larva, Nymph & Adult Mite",
        "symptoms": (
            "Velvety red mites feed on upper citrus leaf surfaces and green fruits, producing "
            "pale grey stippling ('mesophyll collapse'), leaf drop, and dull bronzed rind."
        ),
        "treatments": [
            {
                "name": "Horticultural Mineral Spray Oil (HMSO 1.5%)",
                "active_ingredient": "Paraffinic Horticultural Oil 98%",
                "category": "Organic",
                "efficacy_percent": 85.0,
                "phi_days": 1,
                "cost_per_acre_inr": 560,
                "dosage": "3.0 L in 200 L water per acre",
                "safety_precautions": "Suffocates mite eggs and adults physically; do not mix with sulphur.",
            },
            {
                "name": "Amblyseius (Neoseiulus) californicus Predatory Mites + Hirsutella",
                "active_ingredient": "Hirsutella thompsonii + Predatory Mites",
                "category": "Biological",
                "efficacy_percent": 88.0,
                "phi_days": 0,
                "cost_per_acre_inr": 650,
                "dosage": "1.0 kg Hirsutella WP in 200 L water per acre",
                "safety_precautions": "Avoid broad-spectrum pyrethroids that kill beneficial predatory mites.",
            },
            {
                "name": "Hexythiazox 5.45% EC (Maiden) / Fenpyroximate 5% EC",
                "active_ingredient": "Fenpyroximate 5% EC",
                "category": "Chemical",
                "efficacy_percent": 96.0,
                "phi_days": 14,
                "cost_per_acre_inr": 820,
                "dosage": "200 ml in 200 L water per acre",
                "safety_precautions": "Ensure high-volume wetting of inner and outer citrus tree canopy.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 17 (IP102 ID #85) — CITRUS: Citrus Leafminer (Phyllocnistis citrella)
    # --------------------------------------------------------------------------
    {
        "class_index": 17,
        "ip102_id": 85,
        "crop": "Citrus",
        "common_name": "phyllocnistis citrella stainton",
        "display_name": "Citrus Serpentine Leafminer",
        "scientific_name": "Phyllocnistis citrella",
        "life_stage": "Leaf-Mining Larva & Tiny Moth",
        "symptoms": (
            "Larvae tunnel beneath the epidermis of tender citrus flush leaves, creating "
            "silvery serpentine mines with a central dark frass trail and severe leaf curling."
        ),
        "treatments": [
            {
                "name": "Neem Oil 10,000 ppm + Horticultural Oil Flush Spray",
                "active_ingredient": "Azadirachtin 1% EC",
                "category": "Organic",
                "efficacy_percent": 83.0,
                "phi_days": 2,
                "cost_per_acre_inr": 480,
                "dosage": "500 ml in 200 L water sprayed on new leaf flush",
                "safety_precautions": "Avoid excessive nitrogen fertilizer that triggers continuous vulnerable flush.",
            },
            {
                "name": "Ageniaspis citricola Parasitoid + Spinosad Bio-Derived Spray",
                "active_ingredient": "Spinosad 45% SC (Actinomycete Fermentation)",
                "category": "Biological",
                "efficacy_percent": 90.0,
                "phi_days": 3,
                "cost_per_acre_inr": 740,
                "dosage": "75 ml in 200 L water per acre",
                "safety_precautions": "Highly effective translaminar biological action; spray in evening.",
            },
            {
                "name": "Abamectin 1.9% EC / Imidacloprid 17.8% SL",
                "active_ingredient": "Abamectin 1.9% EC",
                "category": "Chemical",
                "efficacy_percent": 95.0,
                "phi_days": 14,
                "cost_per_acre_inr": 690,
                "dosage": "100 ml in 200 L water per acre",
                "safety_precautions": "Protects wounded leaves from secondary Citrus Canker bacterial infection.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 18 (IP102 ID #96) — MANGO: Mango Hopper (Idioscopus nitidulus)
    # --------------------------------------------------------------------------
    {
        "class_index": 18,
        "ip102_id": 96,
        "crop": "Mango",
        "common_name": "cicadellidae",
        "display_name": "Mango Blossom Leafhopper (Cicadellidae)",
        "scientific_name": "Idioscopus nitidulus / Amritodus atkinsoni",
        "life_stage": "Nymph & Wedge-Shaped Adult Hopper",
        "symptoms": (
            "Nymphs and adults suck sap from tender mango shoots and flower panicles, causing "
            "inflorescence withering, premature fruit drop, and sticky black sooty mold on leaves."
        ),
        "treatments": [
            {
                "name": "Azadirachtin 3000 ppm + Orchard Canopy Pruning",
                "active_ingredient": "Azadirachtin 0.3% EC",
                "category": "Organic",
                "efficacy_percent": 80.0,
                "phi_days": 3,
                "cost_per_acre_inr": 520,
                "dosage": "600 ml in 200 L water per acre at panicle emergence",
                "safety_precautions": "Prune overcrowded branches to allow sunlight penetration.",
            },
            {
                "name": "Metarhizium anisopliae + Verticillium Bio-Consortium",
                "active_ingredient": "Metarhizium anisopliae 1.15% WP",
                "category": "Biological",
                "efficacy_percent": 86.0,
                "phi_days": 1,
                "cost_per_acre_inr": 560,
                "dosage": "1.0 kg in 200 L water sprayed on trunk bark and foliage",
                "safety_precautions": "Safe to use during flowering without harming mango pollinators.",
            },
            {
                "name": "Buprofezin 25% SC / Imidacloprid 17.8% SL (Pre-Bloom)",
                "active_ingredient": "Buprofezin 25% SC",
                "category": "Chemical",
                "efficacy_percent": 96.0,
                "phi_days": 15,
                "cost_per_acre_inr": 740,
                "dosage": "300 ml in 200 L water per acre",
                "safety_precautions": "Do NOT spray chemical insecticides at full open bloom to protect bees.",
            },
        ],
    },
    # --------------------------------------------------------------------------
    # CLASS 19 (IP102 ID #99) — MANGO: Mango Fruit Fly (Bactrocera dorsalis)
    # --------------------------------------------------------------------------
    {
        "class_index": 19,
        "ip102_id": 99,
        "crop": "Mango",
        "common_name": "dacus dorsalis(hendel)",
        "display_name": "Oriental Mango Fruit Fly",
        "scientific_name": "Bactrocera (Dacus) dorsalis",
        "life_stage": "Adult Female Fly & Internal Maggot",
        "symptoms": (
            "Females puncture maturing fruits and lay eggs; dark pin-sized oviposition spots "
            "ooze resinous sap on the peel while maggots cause brown internal pulp rot."
        ),
        "treatments": [
            {
                "name": "Methyl Eugenol Parapheromone Traps + Orchard Sanitation",
                "active_ingredient": "Methyl Eugenol 0.1% Lure",
                "category": "Organic",
                "efficacy_percent": 88.0,
                "phi_days": 0,
                "cost_per_acre_inr": 450,
                "dosage": "10 Methyl Eugenol bottle traps per acre hung at 1.5m height",
                "safety_precautions": "Collect and bury fallen infested fruits 60 cm deep in soil.",
            },
            {
                "name": "Spinosad + Protein Hydrolysate Spot Bait Spray",
                "active_ingredient": "Spinosad 45% SC + Jaggery/Protein Bait",
                "category": "Biological",
                "efficacy_percent": 91.0,
                "phi_days": 3,
                "cost_per_acre_inr": 620,
                "dosage": "50 ml Spinosad + 2 kg Jaggery in 100 L water spot-sprayed on trunks",
                "safety_precautions": "Spot-spray on lower tree trunks and non-fruit foliage only.",
            },
            {
                "name": "Deltamethrin 2.8% EC Pre-Harvest Cover Spray",
                "active_ingredient": "Deltamethrin 2.8% EC",
                "category": "Chemical",
                "efficacy_percent": 94.0,
                "phi_days": 14,
                "cost_per_acre_inr": 540,
                "dosage": "150 ml in 200 L water per acre 21 days before harvest",
                "safety_precautions": "Strictly observe 14-day Pre-Harvest Interval before picking mangoes.",
            },
        ],
    },
]


# ==============================================================================
# HELPER FUNCTIONS TO POPULATE & QUERY OUR CUSTOM 31-BUCKET HASH TABLE
# ==============================================================================

# # [HASH_TABLE] # [DATABASE]
def build_populated_pest_hash_table() -> CustomPestHashTable:
    """
    Creates an instance of `CustomPestHashTable` (31 buckets, polynomial rolling
    hash, linked-list chaining) and inserts all 20 IP102 pest records under:
      1. Composite key: `"<crop>|<common_name>"` (e.g., `"rice|rice leaf roller"`)
      2. Pest common name key: `"<common_name>"` (e.g., `"rice leaf roller"`)
      3. Display name key: `"<display_name>"` (e.g., `"rice leaf roller"`)

    Big-O Time Complexity:
        O(M) where M = 20 pest records (each insertion takes O(1) average time).
    """
    table = CustomPestHashTable()

    # Loop through all 20 IP102 pest records and insert keys into our 31-bucket Hash Table
    for record in IP102_TOP20_PEST_DATABASE:
        crop_key = record["crop"].strip().lower()
        pest_key = record["common_name"].strip().lower()
        display_key = record["display_name"].strip().lower()

        # Primary composite key required by Stage 2: "crop|pest"
        composite_key = f"{crop_key}|{pest_key}"
        bucket_idx = table.insert(composite_key, record)
        # Store the assigned primary hash bucket index inside the record for UI display
        record["hash_bucket"] = bucket_idx

        # Secondary direct keys so lookup also succeeds by pest name alone
        table.insert(pest_key, record)
        table.insert(display_key, record)

    return table


# Instantiate a singleton populated Hash Table ready for O(1) lookups across the app
GLOBAL_PEST_HASH_TABLE: CustomPestHashTable = build_populated_pest_hash_table()
