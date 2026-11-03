"""
================================================================================
FILE: pest_data.py
PURPOSE:
    Stores the complete pest and treatment database as pure Python dictionaries
    for all 20 pests in `config.PEST_SHORTLIST` across the 8 IP102 crops.
    Each record includes:
      - `class_id` (0 to 19, matching its index in `config.PEST_SHORTLIST`)
      - `crop`, `common_name`, `scientific_name`, `symptoms`
      - `treatments`: 3 treatment options (`Organic`, `Biological`, `Chemical`),
        each with `efficacy` (%), `phi_days` (Pre-Harvest Interval in days),
        `cost_inr` (cost per acre in INR), `dosage`, and `safety_notes`.

IMPORTANT AGRONOMIC VERIFICATION NOTE:
    Because the public IP102 dataset only provides insect images and bounding
    boxes (not chemical dosages or costs), all treatment efficacy %, PHI days,
    costs per acre in INR, dosages, and safety precautions below are educational
    reference estimates. They MUST be verified against official ICAR (Indian
    Council of Agricultural Research), TNAU (Tamil Nadu Agricultural University)
    Agritech Portal, or local State Agricultural University package-of-practices
    guidelines before any real field application.

HOW IT CONNECTS TO OTHER FILES:
    - Uses `config.PEST_SHORTLIST` so class indices (0..19) always match.
    - Loaded into the 31-bucket custom Hash Table (`dsa.PestHashTable`) at
      startup and queried by `detect.py` and `app.py`.
================================================================================
"""

# Import typing helpers for clear function and dictionary signatures
from typing import Any, Dict, List

# Import central configuration
import config


# # [DATASET] # [TREATMENT] # [DATABASE]
# List of 20 pest dictionaries in the exact order of `config.PEST_SHORTLIST` (indices 0..19).
# NOTE: Verify all dosages, PHI days, and INR costs against ICAR / TNAU Agritech guidelines.
PEST_DATABASE: List[Dict[str, Any]] = [
    # ==========================================================================
    # CROP 1: RICE (Indices 0, 1, 2)
    # ==========================================================================
    {
        "class_id": 0,
        "crop": "Rice",
        "common_name": "brown plant hopper",
        "scientific_name": "Nilaparvata lugens",
        "symptoms": (
            "Nymphs and adults congregate at the base of rice tillers above water level, "
            "sucking phloem sap and causing circular yellow patches that dry into brownish 'hopperburn'."
        ),
        "treatments": [
            {
                "name": "Neem Seed Kernel Extract (NSKE 5%) + Azadirachtin 1500 ppm",
                "type": "Organic",
                "efficacy": 74.0,  # # [CHANGE_HERE] Verify with ICAR/TNAU rice guidelines
                "phi_days": 3,
                "cost_inr": 420,
                "dosage": "1000 ml per acre in 200 L water directed at tiller base",
                "safety_notes": "Spray in late afternoon; drain excess standing water for 24 hours.",
            },
            {
                "name": "Beauveria bassiana 1.15% WP Entomopathogenic Fungus",
                "type": "Biological",
                "efficacy": 82.0,
                "phi_days": 3,
                "cost_inr": 560,
                "dosage": "1.0 kg per acre in 200 L water",
                "safety_notes": "Do not mix with chemical fungicides; apply when humidity is above 70%.",
            },
            {
                "name": "Pymetrozine 50% WG Systemic Feeding Blocker",
                "type": "Chemical",
                "efficacy": 94.0,
                "phi_days": 14,
                "cost_inr": 890,
                "dosage": "120 g per acre in 200 L water",
                "safety_notes": "Wear nitrile gloves and mask; avoid spraying during bee foraging hours.",
            },
        ],
    },
    {
        "class_id": 1,
        "crop": "Rice",
        "common_name": "asiatic rice borer",
        "scientific_name": "Chilo suppressalis",
        "symptoms": (
            "Larvae bore into rice stems, severing the growing shoot to cause 'deadheart' "
            "during vegetative stage and chaffy white panicles ('whitehead') at heading."
        ),
        "treatments": [
            {
                "name": "Trichogramma japonicum Egg Parasitoid Cards + Neem Oil 2%",
                "type": "Organic",
                "efficacy": 76.0,
                "phi_days": 1,
                "cost_inr": 380,
                "dosage": "5 trichocards (20,000 parasitoids) per acre weekly for 3 weeks",
                "safety_notes": "Staple cards under shaded leaves in morning; do not spray insecticides for 7 days.",
            },
            {
                "name": "Bacillus thuringiensis (Bt) kurstaki 5% WP",
                "type": "Biological",
                "efficacy": 83.0,
                "phi_days": 3,
                "cost_inr": 590,
                "dosage": "400 g per acre in 200 L water",
                "safety_notes": "Apply at evening hours when young larvae hatch before boring into stems.",
            },
            {
                "name": "Chlorantraniliprole 18.5% SC",
                "type": "Chemical",
                "efficacy": 95.0,
                "phi_days": 14,
                "cost_inr": 960,
                "dosage": "60 ml per acre in 200 L water",
                "safety_notes": "Use protective eyewear and mask; do not contaminate irrigation canals.",
            },
        ],
    },
    {
        "class_id": 2,
        "crop": "Rice",
        "common_name": "rice leaf roller",
        "scientific_name": "Cnaphalocrocis medinalis",
        "symptoms": (
            "Greenish caterpillars fold rice leaf blades longitudinally with silk threads "
            "and scrape chlorophyll from inside, leaving papery white longitudinal streaks."
        ),
        "treatments": [
            {
                "name": "Azadirachtin 10000 ppm EC Botanical Spray",
                "type": "Organic",
                "efficacy": 75.0,
                "phi_days": 3,
                "cost_inr": 470,
                "dosage": "400 ml per acre in 200 L water",
                "safety_notes": "Avoid excess nitrogenous urea top-dressing which attracts leaf rollers.",
            },
            {
                "name": "Beauveria bassiana + Bt Kurstaki Bio-Consortium",
                "type": "Biological",
                "efficacy": 84.0,
                "phi_days": 3,
                "cost_inr": 610,
                "dosage": "500 g per acre in 200 L water",
                "safety_notes": "Use clean neutral-pH water and spray thoroughly over upper canopy.",
            },
            {
                "name": "Flubendiamide 20% WG",
                "type": "Chemical",
                "efficacy": 93.0,
                "phi_days": 12,
                "cost_inr": 780,
                "dosage": "50 g per acre in 200 L water",
                "safety_notes": "Wear full sleeves and respirator mask; observe 12-day pre-harvest interval.",
            },
        ],
    },

    # ==========================================================================
    # CROP 2: CORN (Indices 3, 4, 5)
    # ==========================================================================
    {
        "class_id": 3,
        "crop": "Corn",
        "common_name": "corn borer",
        "scientific_name": "Ostrinia furnacalis",
        "symptoms": (
            "Pin-hole shot holes in parallel rows across emerging whorl leaves, sawdust-like "
            "frass at stalk nodes, and broken tassels or tunnelled maize cobs."
        ),
        "treatments": [
            {
                "name": "Trichogramma chilonis Release + Neem Cake Whorl Application",
                "type": "Organic",
                "efficacy": 73.0,
                "phi_days": 2,
                "cost_inr": 410,
                "dosage": "6 tricho-cards per acre + 50 kg neem cake per acre",
                "safety_notes": "Attach cards on the underside of central whorl leaves.",
            },
            {
                "name": "Beauveria bassiana Granular Whorl Placement",
                "type": "Biological",
                "efficacy": 82.0,
                "phi_days": 3,
                "cost_inr": 540,
                "dosage": "1.0 kg per acre mixed with fine sand (1:10) dropped into whorls",
                "safety_notes": "Wear dust mask while applying granules into leaf whorls.",
            },
            {
                "name": "Spinetoram 11.7% SC",
                "type": "Chemical",
                "efficacy": 94.0,
                "phi_days": 10,
                "cost_inr": 920,
                "dosage": "100 ml per acre in 200 L water directed into central whorl",
                "safety_notes": "Toxic to bees during tassel shedding; spray only in early morning or dusk.",
            },
        ],
    },
    {
        "class_id": 4,
        "crop": "Corn",
        "common_name": "armyworm",
        "scientific_name": "Mythimna separata / Spodoptera frugiperda",
        "symptoms": (
            "Ragged window-pane feeding on maize leaves, moist brown frass plugs packed "
            "deep inside the central whorl, and defoliation of young plants."
        ),
        "treatments": [
            {
                "name": "Neem Oil 3000 ppm + Pheromone Funnel Traps",
                "type": "Organic",
                "efficacy": 74.0,
                "phi_days": 3,
                "cost_inr": 460,
                "dosage": "800 ml per acre + 8 pheromone traps per acre",
                "safety_notes": "Apply spray directly into the central leaf whorl where larvae hide.",
            },
            {
                "name": "Spodoptera Nucleopolyhedrovirus (NPV) + Metarhizium anisopliae",
                "type": "Biological",
                "efficacy": 85.0,
                "phi_days": 3,
                "cost_inr": 630,
                "dosage": "250 LE (500 ml) per acre in 200 L water with 0.1% jaggery",
                "safety_notes": "Spray strictly before sunset because UV sunlight degrades viral particles.",
            },
            {
                "name": "Emamectin Benzoate 5% SG",
                "type": "Chemical",
                "efficacy": 95.0,
                "phi_days": 10,
                "cost_inr": 680,
                "dosage": "80 g per acre in 200 L water",
                "safety_notes": "Wear gloves and mask; do not feed treated green fodder to cattle for 10 days.",
            },
        ],
    },
    {
        "class_id": 5,
        "crop": "Corn",
        "common_name": "aphids",
        "scientific_name": "Rhopalosiphum maidis",
        "symptoms": (
            "Dense colonies of dark green to bluish-green soft aphids clustered on tassels "
            "and upper leaves, excreting sticky honeydew that turns black with sooty mould."
        ),
        "treatments": [
            {
                "name": "Potassium Salts of Fatty Acids (Insecticidal Soap 2%) + Neem",
                "type": "Organic",
                "efficacy": 77.0,
                "phi_days": 2,
                "cost_inr": 390,
                "dosage": "1.2 L per acre in 200 L water",
                "safety_notes": "Ensure direct spray contact with aphid colonies on underside of leaves.",
            },
            {
                "name": "Lecanicillium lecanii 1.15% WP Mycoinsecticide",
                "type": "Biological",
                "efficacy": 83.0,
                "phi_days": 3,
                "cost_inr": 520,
                "dosage": "1.0 kg per acre in 200 L water",
                "safety_notes": "Conserves natural ladybird beetle (Coccinellid) predators.",
            },
            {
                "name": "Thiamethoxam 25% WG",
                "type": "Chemical",
                "efficacy": 93.0,
                "phi_days": 14,
                "cost_inr": 540,
                "dosage": "40 g per acre in 200 L water",
                "safety_notes": "Do not apply during active pollen shedding to protect honeybees.",
            },
        ],
    },

    # ==========================================================================
    # CROP 3: WHEAT (Indices 6, 7, 8)
    # ==========================================================================
    {
        "class_id": 6,
        "crop": "Wheat",
        "common_name": "english grain aphid",
        "scientific_name": "Sitobion avenae",
        "symptoms": (
            "Aphids colonize wheat flag leaves and developing earheads during milk/dough "
            "stages, sucking sap and shrivelling developing wheat grains."
        ),
        "treatments": [
            {
                "name": "Azadirachtin 1500 ppm Botanical Foliar Spray",
                "type": "Organic",
                "efficacy": 75.0,
                "phi_days": 3,
                "cost_inr": 410,
                "dosage": "800 ml per acre in 200 L water",
                "safety_notes": "Initiate spray on border rows where aphid migration starts first.",
            },
            {
                "name": "Lecanicillium lecanii Entomopathogenic Fungal Spray",
                "type": "Biological",
                "efficacy": 82.0,
                "phi_days": 3,
                "cost_inr": 510,
                "dosage": "800 g per acre in 200 L water",
                "safety_notes": "Safe for hoverflies and lacewings patrolling wheat earheads.",
            },
            {
                "name": "Flonicamid 50% WG Selective Aphidicide",
                "type": "Chemical",
                "efficacy": 94.0,
                "phi_days": 14,
                "cost_inr": 760,
                "dosage": "60 g per acre in 200 L water",
                "safety_notes": "Observe 14-day pre-harvest interval before wheat grain harvesting.",
            },
        ],
    },
    {
        "class_id": 7,
        "crop": "Wheat",
        "common_name": "red spider",
        "scientific_name": "Petrobia latens / Tetranychus cinnabarinus",
        "symptoms": (
            "Fine silvery-bronze stippling on wheat leaves, delicate webbing under dry "
            "weather, and premature drying of lower and middle foliage."
        ),
        "treatments": [
            {
                "name": "Wettable Sulphur 80% WP (Organic-Approved Miticide)",
                "type": "Organic",
                "efficacy": 79.0,
                "phi_days": 4,
                "cost_inr": 360,
                "dosage": "800 g per acre in 200 L water",
                "safety_notes": "Avoid spraying when daytime temperature exceeds 35°C to prevent leaf scorch.",
            },
            {
                "name": "Hirsutella thompsonii Bio-Acaricide",
                "type": "Biological",
                "efficacy": 83.0,
                "phi_days": 3,
                "cost_inr": 550,
                "dosage": "1.0 L per acre in 200 L water",
                "safety_notes": "Target the lower leaf surface where mites congregate.",
            },
            {
                "name": "Propargite 57% EC Specific Acaricide",
                "type": "Chemical",
                "efficacy": 92.0,
                "phi_days": 14,
                "cost_inr": 720,
                "dosage": "300 ml per acre in 200 L water",
                "safety_notes": "Wear goggles and gloves; avoid skin contact with concentrate.",
            },
        ],
    },
    {
        "class_id": 8,
        "crop": "Wheat",
        "common_name": "wheat blossom midge",
        "scientific_name": "Sitodiplosis mosellana",
        "symptoms": (
            "Tiny orange flies lay eggs on emerging wheat spikes; orange larvae feed on "
            "developing kernels, causing cracked, shrivelled, or missing grains."
        ),
        "treatments": [
            {
                "name": "Neem Oil 3000 ppm + Yellow Sticky Traps at Earhead Height",
                "type": "Organic",
                "efficacy": 73.0,
                "phi_days": 3,
                "cost_inr": 440,
                "dosage": "800 ml per acre + 10 traps per acre",
                "safety_notes": "Apply at early earhead emergence in evening hours.",
            },
            {
                "name": "Beauveria bassiana 1.15% WP",
                "type": "Biological",
                "efficacy": 81.0,
                "phi_days": 3,
                "cost_inr": 530,
                "dosage": "1.0 kg per acre in 200 L water",
                "safety_notes": "Preserves parasitic wasps (Macroglenes penetrans).",
            },
            {
                "name": "Lambda-Cyhalothrin 5% EC",
                "type": "Chemical",
                "efficacy": 91.0,
                "phi_days": 15,
                "cost_inr": 490,
                "dosage": "200 ml per acre in 200 L water",
                "safety_notes": "Wear protective mask and gloves; do not spray in high wind.",
            },
        ],
    },

    # ==========================================================================
    # CROP 4: BEET (Indices 9, 10)
    # ==========================================================================
    {
        "class_id": 9,
        "crop": "Beet",
        "common_name": "beet army worm",
        "scientific_name": "Spodoptera exigua",
        "symptoms": (
            "Gregarious young larvae skeletonize sugarbeet leaves leaving only veins intact, "
            "covered by light webbing; older larvae chew large irregular holes and crown tissue."
        ),
        "treatments": [
            {
                "name": "Azadirachtin 10000 ppm + Pheromone Lure Traps",
                "type": "Organic",
                "efficacy": 76.0,
                "phi_days": 3,
                "cost_inr": 480,
                "dosage": "400 ml per acre + 6 traps per acre",
                "safety_notes": "Target egg masses and first-instar larvae before dispersal.",
            },
            {
                "name": "Spodoptera exigua NPV + Bacillus thuringiensis aizawai",
                "type": "Biological",
                "efficacy": 86.0,
                "phi_days": 2,
                "cost_inr": 640,
                "dosage": "400 g per acre in 200 L water",
                "safety_notes": "Apply in evening for maximum ingestion by actively feeding larvae.",
            },
            {
                "name": "Indoxacarb 14.5% SC",
                "type": "Chemical",
                "efficacy": 94.0,
                "phi_days": 12,
                "cost_inr": 810,
                "dosage": "150 ml per acre in 200 L water",
                "safety_notes": "Wear chemical-resistant gloves; observe 12-day pre-harvest interval.",
            },
        ],
    },
    {
        "class_id": 10,
        "crop": "Beet",
        "common_name": "beet weevil",
        "scientific_name": "Bothynoderes punctiventris / Lixus incanescens",
        "symptoms": (
            "Long-snouted grey-brown adult weevils notch semicircular bites along young "
            "beet leaf margins and sever cotyledon petioles; grubs tunnel into taproots."
        ),
        "treatments": [
            {
                "name": "Perimeter Trench Trapping + Neem Cake Soil Incorporation",
                "type": "Organic",
                "efficacy": 72.0,
                "phi_days": 1,
                "cost_inr": 450,
                "dosage": "80 kg neem cake per acre at sowing",
                "safety_notes": "Check perimeter traps daily during seedling emergence.",
            },
            {
                "name": "Metarhizium anisopliae + Entomopathogenic Nematodes (Steinernema)",
                "type": "Biological",
                "efficacy": 83.0,
                "phi_days": 3,
                "cost_inr": 620,
                "dosage": "1.5 kg per acre soil drench around root zone",
                "safety_notes": "Irrigate lightly before and after application to keep soil moist.",
            },
            {
                "name": "Chlorantraniliprole 18.5% SC + Foliar Protection",
                "type": "Chemical",
                "efficacy": 91.0,
                "phi_days": 14,
                "cost_inr": 890,
                "dosage": "60 ml per acre in 200 L water",
                "safety_notes": "Use personal protective equipment; do not graze livestock on treated tops.",
            },
        ],
    },

    # ==========================================================================
    # CROP 5: ALFALFA (Indices 11, 12, 13)
    # ==========================================================================
    {
        "class_id": 11,
        "crop": "Alfalfa",
        "common_name": "alfalfa weevil",
        "scientific_name": "Hypera postica",
        "symptoms": (
            "Bright green larvae with a white dorsal stripe skeletonize terminal alfalfa "
            "leaf buds, giving heavily infested fields a frosted grey-white appearance."
        ),
        "treatments": [
            {
                "name": "Early First-Cut Harvesting + Azadirachtin 3000 ppm",
                "type": "Organic",
                "efficacy": 78.0,
                "phi_days": 2,
                "cost_inr": 410,
                "dosage": "600 ml per acre in 180 L water",
                "safety_notes": "Early cutting exposes larvae to sunlight and natural Bathyplectes parasitoids.",
            },
            {
                "name": "Beauveria bassiana WP Biological Spray",
                "type": "Biological",
                "efficacy": 82.0,
                "phi_days": 2,
                "cost_inr": 540,
                "dosage": "1.0 kg per acre in 200 L water",
                "safety_notes": "Safe for forage beneficial insects; zero milk withdrawal period.",
            },
            {
                "name": "Indoxacarb 14.5% SC Forage-Safe Regimen",
                "type": "Chemical",
                "efficacy": 93.0,
                "phi_days": 7,
                "cost_inr": 760,
                "dosage": "120 ml per acre in 200 L water",
                "safety_notes": "Wait full 7-day PHI before cutting alfalfa for dairy cattle feed.",
            },
        ],
    },
    {
        "class_id": 12,
        "crop": "Alfalfa",
        "common_name": "tarnished plant bug",
        "scientific_name": "Lygus lineolaris",
        "symptoms": (
            "Adults and green nymphs pierce alfalfa flower buds, blossoms, and immature "
            "seed pods, injecting toxic saliva that causes bud blast and shrivelled seeds."
        ),
        "treatments": [
            {
                "name": "White Sticky Traps + Neem Oil 3000 ppm Spray",
                "type": "Organic",
                "efficacy": 73.0,
                "phi_days": 2,
                "cost_inr": 430,
                "dosage": "800 ml per acre + 8 white sticky traps per acre",
                "safety_notes": "Leave uncut strip refuges to retain native Peristenus parasitoids.",
            },
            {
                "name": "Beauveria bassiana Strain GHA Bio-Insecticide",
                "type": "Biological",
                "efficacy": 81.0,
                "phi_days": 3,
                "cost_inr": 570,
                "dosage": "1.0 kg per acre in 200 L water",
                "safety_notes": "Spray in early evening when Lygus nymphs are active on buds.",
            },
            {
                "name": "Flonicamid 50% WG Selective Feeding Blocker",
                "type": "Chemical",
                "efficacy": 92.0,
                "phi_days": 10,
                "cost_inr": 790,
                "dosage": "60 g per acre in 200 L water",
                "safety_notes": "Low impact on leafcutter bees when applied outside foraging hours.",
            },
        ],
    },
    {
        "class_id": 13,
        "crop": "Alfalfa",
        "common_name": "blister beetle",
        "scientific_name": "Epicauta spp.",
        "symptoms": (
            "Swarms of elongated beetles feed on alfalfa flowers and upper leaves; crushed "
            "beetles in baled hay release cantharidin toxin, which is dangerous to horses and livestock."
        ),
        "treatments": [
            {
                "name": "Swather Conditioner Removal + Pre-Bloom Cutting + Hand Netting",
                "type": "Organic",
                "efficacy": 80.0,
                "phi_days": 0,
                "cost_inr": 320,
                "dosage": "Harvest prior to full bloom without crimping rollers so beetles escape",
                "safety_notes": "Always wear thick gloves; never crush blister beetles on bare skin.",
            },
            {
                "name": "Pyrethrin 0.2% Botanical Knockdown + Beauveria bassiana",
                "type": "Biological",
                "efficacy": 83.0,
                "phi_days": 2,
                "cost_inr": 590,
                "dosage": "600 ml per acre on localized beetle swarms",
                "safety_notes": "Spot-treat aggregated swarms before mowing hay.",
            },
            {
                "name": "Carbaryl 50% WP / Short-PHI Spot Spray",
                "type": "Chemical",
                "efficacy": 91.0,
                "phi_days": 7,
                "cost_inr": 520,
                "dosage": "500 g per acre on infested patches",
                "safety_notes": "Allow beetles to die and drop to ground before baling hay (wait 7 days).",
            },
        ],
    },

    # ==========================================================================
    # CROP 6: VITIS (GRAPE) (Indices 14, 15)
    # ==========================================================================
    {
        "class_id": 14,
        "crop": "Vitis (Grape)",
        "common_name": "ampelophaga",
        "scientific_name": "Ampelophaga rubiginosa (Grape Hornworm)",
        "symptoms": (
            "Large green or brown hornworm caterpillars with diagonal lateral stripes "
            "voraciously defoliate grapevine shoots, stripping entire leaf blades down to petioles."
        ),
        "treatments": [
            {
                "name": "Light Trap Monitoring + Azadirachtin 10000 ppm Spray",
                "type": "Organic",
                "efficacy": 77.0,
                "phi_days": 3,
                "cost_inr": 490,
                "dosage": "400 ml per acre in 200 L water",
                "safety_notes": "Handpick large hornworm larvae during canopy pruning.",
            },
            {
                "name": "Bacillus thuringiensis (Bt) var. kurstaki 5% WP",
                "type": "Biological",
                "efficacy": 88.0,
                "phi_days": 2,
                "cost_inr": 610,
                "dosage": "400 g per acre in 200 L water",
                "safety_notes": "Zero residue on table grapes; highly specific to lepidopteran larvae.",
            },
            {
                "name": "Spinosad 45% SC",
                "type": "Chemical",
                "efficacy": 95.0,
                "phi_days": 7,
                "cost_inr": 1050,
                "dosage": "75 ml per acre in 200 L water",
                "safety_notes": "Observe 7-day pre-harvest interval for table and wine grapes.",
            },
        ],
    },
    {
        "class_id": 15,
        "crop": "Vitis (Grape)",
        "common_name": "brevipoalpus lewisi mcgregor",
        "scientific_name": "Brevipalpus lewisi (Citrus/Grape Flat Mite)",
        "symptoms": (
            "Reddish-brown flat mites feed on grape shoot bases, petioles, and berry "
            "pedicels, causing corky brown scarring on rachises and bronzing of leaves."
        ),
        "treatments": [
            {
                "name": "Micronized Wettable Sulphur 80% WG",
                "type": "Organic",
                "efficacy": 79.0,
                "phi_days": 4,
                "cost_inr": 390,
                "dosage": "800 g per acre in 200 L water",
                "safety_notes": "Do not apply within 14 days of horticultural mineral oil sprays.",
            },
            {
                "name": "Predatory Mite Release (Amblyseius californicus) + Bio-Acaricide",
                "type": "Biological",
                "efficacy": 84.0,
                "phi_days": 1,
                "cost_inr": 680,
                "dosage": "10,000 predatory mites per acre across vine cordons",
                "safety_notes": "Release when morning relative humidity is above 60%.",
            },
            {
                "name": "Spiromesifen 22.9% SC",
                "type": "Chemical",
                "efficacy": 94.0,
                "phi_days": 10,
                "cost_inr": 940,
                "dosage": "160 ml per acre in 200 L water",
                "safety_notes": "Wear gloves and eye protection; thoroughly wet berry clusters and canes.",
            },
        ],
    },

    # ==========================================================================
    # CROP 7: CITRUS (Indices 16, 17)
    # ==========================================================================
    {
        "class_id": 16,
        "crop": "Citrus",
        "common_name": "panonychus citri (citrus red mite)",
        "scientific_name": "Panonychus citri",
        "symptoms": (
            "Velvety dark-red mites puncture epidermal cells on citrus leaves and rind, "
            "producing pale grey stippling ('mesophyll collapse'), leaf drop, and twig dieback."
        ),
        "treatments": [
            {
                "name": "Horticultural Mineral Spray Oil (1.5%) + Neem Extract",
                "type": "Organic",
                "efficacy": 80.0,
                "phi_days": 2,
                "cost_inr": 520,
                "dosage": "1.5 L horticultural oil per acre in 200 L water",
                "safety_notes": "Do not spray on drought-stressed trees or when temperature exceeds 34°C.",
            },
            {
                "name": "Paecilomyces fumosoroseus / Hirsutella Bio-Mite Spray",
                "type": "Biological",
                "efficacy": 84.0,
                "phi_days": 3,
                "cost_inr": 610,
                "dosage": "1.0 L per acre in 200 L water",
                "safety_notes": "Conserves native Phytoseiid predatory mites in citrus orchards.",
            },
            {
                "name": "Hexythiazox 5.45% EC / Etoxazole Miticide",
                "type": "Chemical",
                "efficacy": 94.0,
                "phi_days": 14,
                "cost_inr": 880,
                "dosage": "200 ml per acre in 200 L water",
                "safety_notes": "Rotate acaricide mode of action to prevent mite resistance.",
            },
        ],
    },
    {
        "class_id": 17,
        "crop": "Citrus",
        "common_name": "phyllocnistis citrella (citrus leafminer)",
        "scientific_name": "Phyllocnistis citrella",
        "symptoms": (
            "Silvery serpentine mines with a central dark frass trail on tender new citrus "
            "flush leaves, causing severe leaf curling and predisposing trees to citrus canker."
        ),
        "treatments": [
            {
                "name": "Neem Oil 10000 ppm Flush-Protection Spray + Pheromone Traps",
                "type": "Organic",
                "efficacy": 77.0,
                "phi_days": 3,
                "cost_inr": 480,
                "dosage": "500 ml per acre timed strictly at new flush emergence",
                "safety_notes": "Avoid excessive summer pruning that triggers off-season tender flushes.",
            },
            {
                "name": "Spinosad Botanical-Fermentation Foliar Spray",
                "type": "Biological",
                "efficacy": 86.0,
                "phi_days": 5,
                "cost_inr": 820,
                "dosage": "75 ml per acre in 200 L water",
                "safety_notes": "Safe for Ageniaspis citricola parasitoid wasps once spray dries.",
            },
            {
                "name": "Abamectin 1.9% EC + 0.5% Horticultural Oil Adjuvant",
                "type": "Chemical",
                "efficacy": 95.0,
                "phi_days": 10,
                "cost_inr": 910,
                "dosage": "100 ml per acre in 200 L water",
                "safety_notes": "Wear protective mask and gloves; spray when new flush leaves are 1 cm long.",
            },
        ],
    },

    # ==========================================================================
    # CROP 8: MANGO (Indices 18, 19)
    # ==========================================================================
    {
        "class_id": 18,
        "crop": "Mango",
        "common_name": "dacus dorsalis (mango fruit fly)",
        "scientific_name": "Bactrocera dorsalis (syn. Dacus dorsalis)",
        "symptoms": (
            "Female flies oviposit under the skin of maturing mango fruits, causing resinous "
            "oozing punctures; creamy maggots tunnel through the pulp causing internal rot and fruit drop."
        ),
        "treatments": [
            {
                "name": "Methyl Eugenol Parapheromone Traps + Orchard Sanitation",
                "type": "Organic",
                "efficacy": 83.0,
                "phi_days": 0,
                "cost_inr": 450,
                "dosage": "10 Methyl Eugenol traps per acre hung at 1.5 m height + collect fallen fruit",
                "safety_notes": "Bury all fallen infested mangoes at least 60 cm deep to break pupal cycle.",
            },
            {
                "name": "GF-120 Spinosad Protein Bait Spot-Spray",
                "type": "Biological",
                "efficacy": 89.0,
                "phi_days": 3,
                "cost_inr": 690,
                "dosage": "Spot-spray 50 ml bait solution per tree trunk/foliage weekly",
                "safety_notes": "Do not spray bait directly on fruit; apply as coarse droplets to lower foliage.",
            },
            {
                "name": "Deltamethrin 2.8% EC + 10% Jaggery Bait Spray",
                "type": "Chemical",
                "efficacy": 93.0,
                "phi_days": 12,
                "cost_inr": 640,
                "dosage": "150 ml + 2 kg jaggery per acre in 200 L water at 45 days before harvest",
                "safety_notes": "Observe 12-day pre-harvest interval; wash harvested mangoes in warm water.",
            },
        ],
    },
    {
        "class_id": 19,
        "crop": "Mango",
        "common_name": "cicadellidae (mango leafhopper)",
        "scientific_name": "Idioscopus nitidulus / Amritodus atkinsoni",
        "symptoms": (
            "Wedge-shaped hoppers cluster on mango panicles and tender shoots during "
            "flowering, sucking sap, causing blossom blight/drop, and coating leaves with honeydew."
        ),
        "treatments": [
            {
                "name": "Azadirachtin 3000 ppm Pre-Bloom & Post-Set Spray",
                "type": "Organic",
                "efficacy": 76.0,
                "phi_days": 3,
                "cost_inr": 460,
                "dosage": "700 ml per acre in 200 L water",
                "safety_notes": "Prune dense interlocking branches to allow sunlight penetration.",
            },
            {
                "name": "Metarhizium anisopliae + Verticillium lecanii Bio-Spray",
                "type": "Biological",
                "efficacy": 84.0,
                "phi_days": 3,
                "cost_inr": 590,
                "dosage": "1.0 kg per acre in 200 L water directed at tree trunk and panicles",
                "safety_notes": "Safe for syrphid flies and honeybees visiting mango blossoms.",
            },
            {
                "name": "Imidacloprid 17.8% SL / Buprofezin 25% SC",
                "type": "Chemical",
                "efficacy": 95.0,
                "phi_days": 15,
                "cost_inr": 670,
                "dosage": "80 ml per acre in 200 L water (at panicle emergence & pea-size fruit)",
                "safety_notes": "NEVER spray chemical insecticides during peak 50% open mango flowering.",
            },
        ],
    },
]


# # [DATASET] Helper function to look up a pest dictionary by its integer class_id (0..19)
def get_pest_by_class_id(class_id: int) -> Dict[str, Any]:
    """
    Returns the pest dictionary for `class_id` (0..19).
    Since `PEST_DATABASE` is ordered identically to `config.PEST_SHORTLIST`,
    index `class_id` directly gives the matching record.
    """
    if 0 <= class_id < len(PEST_DATABASE):
        return PEST_DATABASE[class_id]
    # Fallback to index 0 if out of range
    return PEST_DATABASE[0]
