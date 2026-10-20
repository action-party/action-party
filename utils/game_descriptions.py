"""
Per-game text descriptions for all 46 Melting Pot mini-map games.

Each description focuses on what is visually on screen: the environment,
objects, sprites, player appearance, and observable actions. No game theory,
strategy, or reward logic -- only what you would see in the rendered frames.

Colors are pastel throughout (baked into the meltingpot package).
Player sprites have distinct pastel body colors and pure black eyes.
"""

GAME_DESCRIPTIONS = {
    # =========================================================================
    # ALLELOPATHIC HARVEST
    # =========================================================================
    "allelopathic_harvest__open": (
        "Pastel-colored player sprites walk across a dark gray field covered "
        "with small red, green, and blue berry patches. Players fire colored "
        "planting beams that change berry colors and turn the player's body "
        "to match. Players walk over ripe berries to consume them."
    ),

    # =========================================================================
    # BACH OR STRAVINSKY IN THE MATRIX
    # =========================================================================
    "bach_or_stravinsky_in_the_matrix__arena": (
        "Multiple player sprites move inside a dark walled arena with gray "
        "brick borders. Mint and pink square resource tiles are arranged in "
        "columns on the floor. Players walk over tiles to collect them and "
        "fire interaction beams at each other."
    ),
    "bach_or_stravinsky_in_the_matrix__repeated": (
        "Two player sprites move inside a small dark walled room with gray "
        "brick borders. Mint and pink square resource tiles sit on the floor. "
        "Players walk over tiles to collect them and fire interaction beams "
        "at each other."
    ),

    # =========================================================================
    # CHEMISTRY
    # =========================================================================
    "chemistry__three_metabolic_cycles": (
        "Player sprites move across a white open field scattered with colorful "
        "arrow-shaped molecule sprites in blue, green, red, and cyan. Players "
        "walk into molecules to pick them up and carry them. Molecules near "
        "each other react and transform into different colored molecules."
    ),
    "chemistry__three_metabolic_cycles_with_plentiful_distractors": (
        "Player sprites move across a white open field with many colorful "
        "arrow-shaped molecule sprites in blue, green, red, cyan, and additional "
        "distractor molecules. Players walk into molecules to pick them up "
        "and carry them. Nearby molecules react and change color."
    ),
    "chemistry__two_metabolic_cycles": (
        "Player sprites move across a white open field scattered with colorful "
        "arrow-shaped molecule sprites in blue, green, red, and cyan. Players "
        "walk into molecules to pick them up and carry them. Molecules near "
        "each other react and transform into different colored molecules."
    ),
    "chemistry__two_metabolic_cycles_with_distractors": (
        "Player sprites move across a white open field with colorful arrow-shaped "
        "molecule sprites and additional distractor molecules. Players walk into "
        "molecules to pick them up and carry them. Nearby molecules react and "
        "change color."
    ),

    # =========================================================================
    # CHICKEN IN THE MATRIX
    # =========================================================================
    "chicken_in_the_matrix__arena": (
        "Multiple player sprites move inside a dark walled arena with gray "
        "brick borders. Mint and pink square resource tiles are arranged in "
        "columns on the floor. Players walk over tiles to collect them and "
        "fire interaction beams at each other."
    ),
    "chicken_in_the_matrix__repeated": (
        "Two player sprites move inside a small dark walled room with gray "
        "brick borders. Mint and pink square resource tiles sit on the floor. "
        "Players walk over tiles to collect them and fire interaction beams "
        "at each other."
    ),

    # =========================================================================
    # CLEAN UP
    # =========================================================================
    "clean_up": (
        "Player sprites move across a map split into a sandy upper area and a "
        "green lower area with a blue-green polluted river section between them. "
        "Gray brick walls surround the map. Pink apple sprites appear on the "
        "green grass. Players fire cleaning beams at the river pollution."
    ),

    # =========================================================================
    # COINS
    # =========================================================================
    "coins": (
        "Two player sprites move around a black-floored room surrounded by gray "
        "brick walls. Small pink and yellow coin sprites periodically appear at "
        "random positions on the floor. Players walk over coins to pick them up "
        "and coins disappear on contact."
    ),

    # =========================================================================
    # COLLABORATIVE COOKING
    # =========================================================================
    "collaborative_cooking__asymmetric": (
        "Two player sprites move on dark walkway tiles inside an asymmetric "
        "kitchen made of brown counter tiles. Small red tomato sprites sit on "
        "dispensers, gray cooking pots bubble on counters, and white plate "
        "sprites are available. Players carry items between stations."
    ),
    "collaborative_cooking__circuit": (
        "Two player sprites move on dark tiles around a central brown counter "
        "forming a circuit-shaped kitchen. Red tomato sprites, gray cooking "
        "pots, and white plate sprites sit on the counters. Players pick up "
        "items and carry them between stations along the circuit."
    ),
    "collaborative_cooking__cramped": (
        "Two player sprites move in a very small kitchen made of brown counter "
        "tiles with minimal dark floor space. Red tomato sprites, gray cooking "
        "pots, and white plate sprites sit on counters. Players squeeze past "
        "each other carrying items between stations."
    ),
    "collaborative_cooking__crowded": (
        "Five player sprites move in a large kitchen split by a central brown "
        "counter with a narrow dark passage. Red tomato sprites, gray cooking "
        "pots, and white plate sprites sit on counters. Players navigate the "
        "passage and carry items between kitchen stations."
    ),
    "collaborative_cooking__figure_eight": (
        "Six player sprites move through narrow dark corridors in a figure-eight "
        "shaped kitchen with brown counters and small alcoves. Red tomato "
        "sprites, gray cooking pots, and white plate sprites sit on counters. "
        "Players duck into alcoves to let others pass."
    ),
    "collaborative_cooking__forced": (
        "Two player sprites move on opposite sides of a brown counter dividing "
        "a small kitchen. One side has red tomato and white plate dispensers, "
        "the other has gray cooking pots and a delivery point. Players place "
        "items on the shared counter for the other to pick up."
    ),
    "collaborative_cooking__ring": (
        "Two player sprites loop around a ring-shaped kitchen with brown "
        "counters lining both sides of a narrow dark corridor. Red tomato "
        "sprites, gray cooking pots, and white plate sprites sit on counters. "
        "Players travel in the same direction around the ring."
    ),

    # =========================================================================
    # COMMONS HARVEST
    # =========================================================================
    "commons_harvest__closed": (
        "Player sprites move through green-floored rooms connected by narrow "
        "corridors, surrounded by gray brick walls. Pink apple sprites grow on "
        "the green floor tiles. Players walk over apples to consume them. "
        "Players fire zapping beams that temporarily remove other players."
    ),
    "commons_harvest__open": (
        "Player sprites move across an open sandy field with green patches, "
        "surrounded by gray brick walls. Pink apple sprites grow on the green "
        "tiles. Players walk over apples to consume them and new apples slowly "
        "regrow nearby. Players fire zapping beams at each other."
    ),
    "commons_harvest__partnership": (
        "Player sprites move through green-floored rooms with two entrances "
        "each, connected by corridors and surrounded by gray brick walls. Pink "
        "apple sprites grow on the green floor. Players walk over apples to "
        "consume them and fire zapping beams at each other."
    ),

    # =========================================================================
    # COOP MINING
    # =========================================================================
    "coop_mining": (
        "Player sprites move across a dark floor with gray brick walls and "
        "scattered gray and yellow ore blocks. Gray iron ore blocks and bright "
        "yellow gold ore blocks appear randomly in empty spaces. Players walk "
        "next to ore and fire mining beams to extract it."
    ),

    # =========================================================================
    # DAYCARE
    # =========================================================================
    "daycare": (
        "Two player sprites move across a brown earthy ground with gray stone "
        "borders. Green trees of varying heights and small green shrubs bearing "
        "fruit are scattered across the field. Players walk up to trees and "
        "shrubs to harvest fruit from them."
    ),

    # =========================================================================
    # EXTERNALITY MUSHROOMS
    # =========================================================================
    "externality_mushrooms__dense": (
        "Player sprites move across a dark olive-green field with a wooden "
        "fence border. Small pastel mushroom sprites in pink, green, cyan, "
        "and orange are scattered across the ground. Players walk over "
        "mushrooms to eat them and new mushrooms regrow over time."
    ),

    # =========================================================================
    # FACTORY COMMONS
    # =========================================================================
    "factory_commons__either_or": (
        "Player sprites move across a light purple-gray factory floor surrounded "
        "by dark blue-gray walls. Gray machine sprites with orange hatched input "
        "and output bins sit in the corners. Small cyan cube sprites and pink "
        "apple sprites appear near machines. Players carry cubes to machines."
    ),

    # =========================================================================
    # FRUIT MARKET
    # =========================================================================
    "fruit_market__concentric_rivers": (
        "Player sprites move across a sandy field with wooden fence borders and "
        "concentric rectangular cyan river rings. Green and red fruit tree "
        "sprites with visible fruit grow between the rivers. Players walk up "
        "to trees to harvest fruit and approach each other to trade."
    ),

    # =========================================================================
    # GIFT REFINEMENTS
    # =========================================================================
    "gift_refinements": (
        "Player sprites move across a dark floor with gray brick walls and "
        "scattered small colored token sprites. Players walk over tokens to "
        "pick them up and fire gifting beams at other players to transfer "
        "tokens. Gifted tokens split into multiple smaller tokens."
    ),

    # =========================================================================
    # PAINTBALL
    # =========================================================================
    "paintball__capture_the_flag": (
        "Two teams of player sprites move through a gray-walled arena bordered "
        "by purple indicator tiles. Players fire beams that paint the floor red "
        "or blue in large patches. Small flag sprites sit at each team's base. "
        "Players pick up the opposing flag and carry it back."
    ),
    "paintball__king_of_the_hill": (
        "Two teams of player sprites move through a gray-walled arena bordered "
        "by purple indicator tiles. Players fire beams that paint the floor red "
        "or blue in large patches. A central hill area changes color as teams "
        "paint over it. Purple indicator tiles show control status."
    ),

    # =========================================================================
    # PREDATOR PREY
    # =========================================================================
    "predator_prey__alley_hunt": (
        "Predator and prey player sprites navigate narrow gray-walled alleys "
        "with dead ends. Small green apple sprites and brown acorn sprites sit "
        "on the floor. Green grass patches block some paths. Colored stamina "
        "bars appear above each player sprite."
    ),
    "predator_prey__open": (
        "Predator and prey player sprites move across a large open field with "
        "rows of small green apple sprites and brown acorn sprites. Tall green "
        "grass borders the field edges. Colored stamina bars appear above each "
        "player. Predators chase prey across the open space."
    ),
    "predator_prey__orchard": (
        "Predator and prey player sprites move in an environment with green "
        "apple sprites in the upper area and brown acorn sprites near tall "
        "green grass on the sides. Players navigate between food areas and "
        "grass patches. Colored stamina bars appear above each player."
    ),
    "predator_prey__random_forest": (
        "Predator and prey player sprites move through randomly scattered "
        "patches of tall green grass on a light floor. Brown acorn sprites sit "
        "on the ground between grass patches. Colored stamina bars appear above "
        "each player. Predators navigate around grass to chase prey."
    ),

    # =========================================================================
    # PRISONERS DILEMMA IN THE MATRIX
    # =========================================================================
    "prisoners_dilemma_in_the_matrix__arena": (
        "Multiple player sprites move inside a dark walled arena with gray "
        "brick borders. Mint and pink square resource tiles are arranged in "
        "columns on the floor. Players walk over tiles to collect them and "
        "fire interaction beams at each other."
    ),
    "prisoners_dilemma_in_the_matrix__repeated": (
        "Two player sprites move inside a small dark walled room with gray "
        "brick borders. Mint and pink square resource tiles sit on the floor. "
        "Players walk over tiles to collect them and fire interaction beams "
        "at each other."
    ),

    # =========================================================================
    # PURE COORDINATION IN THE MATRIX
    # =========================================================================
    "pure_coordination_in_the_matrix__arena": (
        "Multiple player sprites move inside a dark walled arena with gray "
        "brick borders. Soft red, soft green, and soft blue square resource "
        "tiles are arranged in columns on the floor. Players walk over tiles "
        "to collect them and fire interaction beams at each other."
    ),
    "pure_coordination_in_the_matrix__repeated": (
        "Two player sprites move inside a small dark walled room with gray "
        "brick borders. Soft red, soft green, and soft blue square resource "
        "tiles sit on the floor. Players walk over tiles to collect them "
        "and fire interaction beams at each other."
    ),

    # =========================================================================
    # RATIONALIZABLE COORDINATION IN THE MATRIX
    # =========================================================================
    "rationalizable_coordination_in_the_matrix__arena": (
        "Multiple player sprites move inside a dark walled arena with gray "
        "brick borders. Soft red, soft green, and soft blue square resource "
        "tiles are arranged in columns on the floor. Players walk over tiles "
        "to collect them and fire interaction beams at each other."
    ),
    "rationalizable_coordination_in_the_matrix__repeated": (
        "Two player sprites move inside a small dark walled room with gray "
        "brick borders. Soft red, soft green, and soft blue square resource "
        "tiles sit on the floor. Players walk over tiles to collect them "
        "and fire interaction beams at each other."
    ),

    # =========================================================================
    # RUNNING WITH SCISSORS IN THE MATRIX
    # =========================================================================
    "running_with_scissors_in_the_matrix__arena": (
        "Multiple player sprites move inside a dark walled arena with gray "
        "brick borders. Yellow, violet, and cyan square resource tiles are "
        "arranged in columns on the floor. Players walk over tiles to collect "
        "them and fire interaction beams at each other."
    ),
    "running_with_scissors_in_the_matrix__one_shot": (
        "Two player sprites move inside a small dark walled room with gray "
        "brick borders. Yellow, violet, and cyan square resource tiles sit "
        "on the floor. Players walk over tiles to collect them and fire an "
        "interaction beam at each other once."
    ),
    "running_with_scissors_in_the_matrix__repeated": (
        "Two player sprites move inside a small dark walled room with gray "
        "brick borders. Yellow, violet, and cyan square resource tiles sit "
        "on the floor. Players walk over tiles to collect them and fire "
        "interaction beams at each other."
    ),

    # =========================================================================
    # STAG HUNT IN THE MATRIX
    # =========================================================================
    "stag_hunt_in_the_matrix__arena": (
        "Multiple player sprites move inside a dark walled arena with gray "
        "brick borders. Mint and pink square resource tiles are arranged in "
        "columns on the floor. Players walk over tiles to collect them and "
        "fire interaction beams at each other."
    ),
    "stag_hunt_in_the_matrix__repeated": (
        "Two player sprites move inside a small dark walled room with gray "
        "brick borders. Mint and pink square resource tiles sit on the floor. "
        "Players walk over tiles to collect them and fire interaction beams "
        "at each other."
    ),

    # =========================================================================
    # TERRITORY
    # =========================================================================
    "territory__inside_out": (
        "Five player sprites start at the edges of a dark background and move "
        "inward toward a central maze of gray brick resource walls. Players "
        "paint walls in their unique pastel color using a brush. Painted walls "
        "brighten after drying. Players fire zapping beams to destroy walls."
    ),
    "territory__rooms": (
        "Player sprites start in separate dark rooms divided by gray brick "
        "resource walls. Each player paints nearby walls in their unique "
        "pastel color using a brush, and dried paint appears brighter. "
        "Players fire zapping beams to break through walls or hit others."
    ),
}


def get_game_description(game_name: str) -> str:
    """Get the text description for a game.

    Falls back to a generic description if game not found.
    """
    if game_name in GAME_DESCRIPTIONS:
        return GAME_DESCRIPTIONS[game_name]

    # Fallback: generate a basic description from the game name
    readable = game_name.replace('__', ': ').replace('_', ' ').title()
    return f"Player sprites move and interact in a multiplayer grid game called {readable}."
