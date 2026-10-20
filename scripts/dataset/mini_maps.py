"""
Mini map variants for Melting Pot games to fit within 16x16 grid.

At 4x scale with 8px native sprites, 16x16 grid = 512x512 pixels with 32px sprites.

Usage:
    from mini_maps import get_mini_map, get_mini_player_count, MINI_GAMES_512
    apply_mini_config(game_name, config)
"""


def _make_map(lines, target_w=16):
    """Build a map string, padding each line with spaces to target_w if needed.
    Raises ValueError if any line exceeds target_w."""
    result = []
    for i, line in enumerate(lines):
        if len(line) > target_w:
            raise ValueError(f"Line {i} has {len(line)} chars > {target_w}: {line!r}")
        # Pad short interior lines by inserting space before the last wall char
        if len(line) < target_w:
            diff = target_w - len(line)
            # Insert padding before the last character (wall)
            line = line[:-1] + ' ' * diff + line[-1]
        result.append(line)
    return '\n'.join(result)


# =============================================================================
# TIER 1: Already fit or trivial shrink
# =============================================================================

MINI_DAYCARE = _make_map([
    "/______________+",
    "!~~~~~~~~~~~~~~|",
    "!~~~~~~~~~~~~~~|",
    "!~~~~~~~~~~~~~~|",
    "!~~~~~~~~~~~~~~|",
    "!~~~~~PPP~~~~~~|",
    "!~~~~~PPP~~~~~~|",
    "!~~~~~PPP~~~~~~|",
    "!~~~~~~~~~~~~~~|",
    "!~~~~~~~~~~~~~~|",
    "!~~~~~~~~~~~~~~|",
    "!~~~~~~~~~~~~~~|",
    "(--------------)"])

MINI_EXTERNALITY_MUSHROOMS_DENSE = _make_map([
    "/______________+",
    "'##############`",
    "!              |",
    "! R        G   |",
    "!     R        |",
    "!          G   |",
    "!   B    O     |",
    "!            B |",
    "!     R        |",
    "!              |",
    "!   B      G   |",
    "!              |",
    "!              |",
    "(--------------)"])

# =============================================================================
# TIER 2A: In-the-matrix games (shared templates)
# =============================================================================

MINI_MATRIX_ARENA = _make_map([
    "WWWWWWWWWWWWWWWW",
    "WPP  W  W  PPPW",
    "WPP           PW",
    "W              W",
    "W   11    22   W",
    "W   1a    22   W",
    "W W 1a  W aa   W",
    "WW  aa    11  WW",
    "W   22    1a   W",
    "W   22  W 1a W W",
    "W   aa    aa   W",
    "W         22   W",
    "W              W",
    "WP           PPW",
    "WPP  W  W  PPPW",
    "WWWWWWWWWWWWWWWW"])

MINI_MATRIX_REPEATED = _make_map([
    "WWWWWWWWWWWWWWWW",
    "Wn      n     nW",
    "W  WW W  W WW  W",
    "W   W11aWa22W  W",
    "Wn  W11aWa22W nW",
    "W    11a  a22  W",
    "W              W",
    "Wn  WW  n WW  nW",
    "W              W",
    "W    22a  a11  W",
    "Wn  W22aWa11W nW",
    "W   W22aWa11W  W",
    "W  WW W  W WW  W",
    "Wn      n     nW",
    "WWWWWWWWWWWWWWWW"])

MINI_RWS_ONE_SHOT = _make_map([
    "WWWWWWWWWWWWWWWW",
    "Wn r  a  a p  nW",
    "W              W",
    "Wn r  a  a p  nW",
    "W              W",
    "Wn r  a  a p  nW",
    "W              W",
    "W   n    n   n W",
    "W              W",
    "Wn s  a  a a  nW",
    "W              W",
    "Wn s  a  a a  nW",
    "W              W",
    "Wn s  a  a a  nW",
    "WWWWWWWWWWWWWWWW"])

# =============================================================================
# TIER 2B: Chemistry games (shared template)
# =============================================================================

MINI_CHEMISTRY_TWO = _make_map([
    "~~~~~~~~a~~~~~~~",
    "~~~~~~c~~~~~~~~~",
    "~~~~~~~~b~~~~~~~",
    "~~~~~~~~~~~~~~~~",
    "~~~~~~~~~~~~~1~~",
    "~~~~~~~~~~~~~~~~",
    "1~3~~hhhhh~~3~2~",
    "~~~~~~~~~~~~~~~~",
    "~2~~~~~~~~~~~~~~",
    "~~~~~~~~~~~~~~~~",
    "~~~~~c~~~~~~~~~~",
    "~~~~~~a~~~~~~~~~",
    "~~~~~b~~~~~~~~~~",
    "~~~~~~~~~~~~~~~~"])

MINI_CHEMISTRY_THREE = _make_map([
    "~~~~~~~~a~~~~~~~",
    "~~~~~~c~~~~~~~~~",
    "~~~~~~~~b~~~~~~~",
    "~~~~~~~~~~~~~~~~",
    "~~~~~~~~~~~~~1~~",
    "~~~~~~~~~~~~~~~~",
    "1~3~~hhhhh~~3~2~",
    "~~~~~~~~~~~~~~~~",
    "~2~~~~~~~~~~~~~~",
    "~~~~~~~~~~~~~~~~",
    "~~~~~c~~~~~4~~6~",
    "~~~~~~a~~~~~~~~~",
    "~~~~~b~~~~~~~5~~",
    "~~~~~~~~~~~~~~~~"])

# =============================================================================
# TIER 2C: Commons harvest
# =============================================================================

MINI_COMMONS_HARVEST_OPEN = _make_map([
    "WWWWWWWWWWWWWWWW",
    "WAA   A  A   AAW",
    "WA  AAA  AAA  AW",
    "W   AAA  AAA   W",
    "W    A    A    W",
    "W A    Q     A W",
    "WAAAQ       QAAW",
    "W AA        AA W",
    "W  A        A  W",
    "W              W",
    "W PPPPPPPPPPPP W",
    "WPPPPPPPPPPPPPPW",
    "WPPPPPPPPPPPPPPW",
    "WWWWWWWWWWWWWWWW"])

MINI_COMMONS_HARVEST_CLOSED = _make_map([
    "WWWWWWWWWWWWWWWW",
    "WAA  A WW A  AAW",
    "WA  AAAWWAAA  AW",
    "W   AAA WW AA  W",
    "W    A  WW  A  W",
    "W A  Q  WW Q A W",
    "WAAA    WW  AAAW",
    "W AA WWWWWW AA W",
    "W  A    WW  A  W",
    "WWWWWW  WW WWWWW",
    "W       WW     W",
    "W PPPPPPPPPPPP W",
    "WPPPPPPPPPPPPPPW",
    "WPPPPPPPPPPPPPPW",
    "WPPPPPPPPPPPPPPW",
    "WWWWWWWWWWWWWWWW"])

MINI_COMMONS_HARVEST_PARTNERSHIP = _make_map([
    "WWWWWWWWWWWWWWWW",
    "WAA  A II A  AAW",
    "WA  AAAIIAAA  AW",
    "W   AAA II AA  W",
    "W    A  II  A  W",
    "W A  Q  WW Q A W",
    "WAAA    II  AAAW",
    "W AA WWWWWW AA W",
    "W  A    WW  A  W",
    "WWWWWW  WW WWWWW",
    "W       WW     W",
    "W PPPPPPPPPPPP W",
    "WPPPPPPPPPPPPPPW",
    "WPPPPPPPPPPPPPPW",
    "WPPPPPPPPPPPPPPW",
    "WWWWWWWWWWWWWWWW"])

# =============================================================================
# TIER 2C: Territory
# =============================================================================

# territory__rooms: 2x2 rooms (TORUS topology), 14x14 is fine (pads to 16)
# But for TORUS, all rows must be same width. Use 16x14.
MINI_TERRITORY_ROOMS = _make_map([
    "JRRRRRRRLJRRRRRL",
    "R,,,,,,,RR,,,,,R",
    "R,,,,,,,RR,,,,,R",
    "R,,,,P,,RR,,P,,R",
    "R,,,,,,,RR,,,,,R",
    "R,,,,,,,RR,,,,,R",
    "TRRRRRRRLTRRRRRF",
    "JRRRRRRRLJRRRRRL",
    "R,,,,,,,RR,,,,,R",
    "R,,,,,,,RR,,,,,R",
    "R,,,,P,,RR,,P,,R",
    "R,,,,,,,RR,,,,,R",
    "R,,,,,,,RR,,,,,R",
    "TRRRRRRRLTRRRRRF"], target_w=16)
# Note: territory rooms uses TORUS so edges wrap. 16 wide = 512px at 4x.

MINI_TERRITORY_INSIDE_OUT = _make_map([
    "F==============T",
    "|,,,,,,P,,,,,,,|",
    "|,P,,QQ,,,QQ,,P|",
    "|,,RRR,,RRR,,,,|",
    "|,,R,RAAR,RAA,,|",
    "|,,RRR,BRRR,B,,|",
    "|,Q,ABBRABBR,Q,|",
    "|,Q,ABBRABBR,Q,|",
    "|,,RRR,BRRR,B,,|",
    "|,,R,RAAR,RAA,,|",
    "|,,RRR,,RRR,,,,|",
    "|,Q,ABBRABBR,Q,|",
    "|,Q,ABBRABBR,Q,|",
    "|,P,,QQ,,,QQ,,P|",
    "|,,,,,,P,,,,,,,|",
    "L==============J"])

# =============================================================================
# TIER 2C: Paintball
# =============================================================================

MINI_PAINTBALL_CTF = _make_map([
    "IIIIIIIIIIIIIIII",
    "IWWWWWWWWWWWWWWI",
    "IWPP,,F,PP,PPWWI",
    "IWP,,,,,,,,,PWWI",
    "IWHWW,,WWW,WHHWI",
    "IWHW,,D,,D,,WHWI",
    "IW,,,,,,,,,,,,WI",
    "IW,,,WW,,WW,,,WI",
    "IW,,,WW,,WW,,,WI",
    "IW,,,,,,,,,,,,WI",
    "IWQW,,D,,D,,WQWI",
    "IWQWW,,WWW,WQQWI",
    "IWQ,,,,,,,,,QWWI",
    "IWQQ,,G,QQ,QQWWI",
    "IWWWWWWWWWWWWWWI",
    "IIIIIIIIIIIIIIII"])

MINI_PAINTBALL_KOTH = _make_map([
    "IIIIIIIIIIIIIIII",
    "IWWWWWWWWWWWWWWI",
    "IWPP,,P,PP,PPWWI",
    "IWP,,,,,,,,,PWWI",
    "IW,,,WDWDWW,,WI",
    "IW,,,uuuuuuu,,WI",
    "IWHW,lGGGGGr,HWI",
    "IWHW,lGGIGGr,HWI",
    "IWHW,lGGGGGr,HWI",
    "IW,,,ddddddd,,WI",
    "IW,,,WWDWDW,,WI",
    "IWQ,,,,,,,,,QWWI",
    "IWQQ,,Q,QQ,QQWWI",
    "IWWWWWWWWWWWWWWI",
    "IIIIIIIIIIIIIIII",
    "IIIIIIIIIIIIIIII"])

# =============================================================================
# TIER 2C: Factory commons
# =============================================================================

MINI_FACTORY_COMMONS = _make_map([
    ";______________,",
    "!      c       |",
    "!     cCc      |",
    "! ls   C   lt  |",
    "! Oj       O#  |",
    "!  z        z  |",
    "!  x        x  |",
    "!     cCc      |",
    "!     cCc      |",
    "! ls       lt  |",
    "! Oj       O#  |",
    "!  z    C   z  |",
    "!     cCc      |",
    "________________"])

# =============================================================================
# TIER 2C: Gift refinements
# =============================================================================

MINI_GIFT_REFINEMENTS = _make_map([
    "WWWWWWWWWWWWWWWW",
    "WTTTTTTTTTTTTTTW",
    "WTPTTPTTTPTPTTW",
    "WTTTTTWTTTTTTTTW",
    "WTTTTTWTTTTWTTTW",
    "WTPTWWTTTTTWTPTW",
    "WTTTTTTWTTPTTTTW",
    "WTTTTTTWTTTTWTTW",
    "WTPTTTWTTTTWTPTW",
    "WTTTTTWTTTTTTTTW",
    "WTTTTTWTTTTWTTTW",
    "WTTTTTWTTTTTTTTW",
    "WTTTTTWTTTWWWTTW",
    "WTPTTPTTTPTPTTW",
    "WTTTTTTTTTTTTTTW",
    "WWWWWWWWWWWWWWWW"])

# =============================================================================
# TIER 2C: Coop mining
# =============================================================================

MINI_COOP_MINING = _make_map([
    "WWWWWWWWWWWWWWWW",
    "WOOOOOOOOOOOOOOW",
    "WOPOOOOOPOOOOOPW",
    "WOOOOOWOOOOOOOOW",
    "WOOOOOWOOOOWOOW",
    "WOOOOOOWWWOOOPOW",
    "WOPOWOOOOOOOOOW",
    "WOOOOOWOOPOOOOOW",
    "WOOOOOWOOOOWOOW",
    "WOOOOOOOOOOOWOOW",
    "WOPOOOWWWWOOOOW",
    "WOOWWWOOOOOOOOW",
    "WOOOOWOOOPOOOOOW",
    "WOPOOOOOPOOOOOPW",
    "WOOOOOOOOOOOOOOW",
    "WWWWWWWWWWWWWWWW"])

# =============================================================================
# TIER 3: Hard games
# =============================================================================

MINI_CLEAN_UP = _make_map([
    "WWWWWWWWWWWWWWWW",
    "WHFHFHFHFHHFFFHW",
    "WHFHFHFHFHHFFFHW",
    "WHFHFHFHFHHFFFHW",
    "W========+~FHHfW",
    "W P  P   =+~SfPW",
    "W  P  P   <~S> W",
    "W P  P    <~S>PW",
    "W  P     P<~S> W",
    "W^T^T^T^T^;~S,TW",
    "WBBBBBBBBBBssBBW",
    "WBBBBBBBBBBBBBBW",
    "WBBBBBBBBBBBBBBW",
    "WWWWWWWWWWWWWWWW"])

MINI_PREDATOR_PREY_OPEN = _make_map([
    "/;____________,/",
    ";]************[,",
    "!vvvvvvvvvvvvvv|",
    "!''''''''''''''|",
    "!''XXXXXXXXXX''|",
    "!''Xaaaa&aaaX''|",
    "!'aaaaaAaaAaaa'|",
    "!AaaaaaaaaaaaAa|",
    "!aAaaaaaaa&aaaa|",
    "!'aaAaaaaaaAa''|",
    "!''XaaaaaaaaX''|",
    "!''XXXXXXXXXX''|",
    "!''''''''''''''|",
    "!^^^^^^^^^^^^^^|",
    "L+************=J",
    "/L~~~~~~~~~~~~J/"])

MINI_PREDATOR_PREY_ALLEY = _make_map([
    ";______________,",
    "!a''''''''''aa|",
    "!''=~~+'=+''''|",
    "!''|//!a|!''=+|",
    "!''[_,!a[]''|!|",
    "!'''a|!a''X'[]|",
    "!a=~J!a''X'''a|",
    "!a|/![___]'''a|",
    "!'[_]'XX''=~~J|",
    "!''''X''a<*[_,|",
    "!''X'''XX<***||",
    "!a'''a'X'<***||",
    "!'X'''XX'<*=~J|",
    "!a|/!X''a=~J;,|",
    "!'[,!'=~+'|/!a|",
    "L~~~~~~~~~~~~~~J"])

MINI_PREDATOR_PREY_ORCHARD = _make_map([
    "/;____________,/",
    ";]a'X'XX'a''A[,",
    "!a'aaAaaa'aaa'|",
    "!'&'aaaa'Aaa'a|",
    "!a'X''X'aaa''a|",
    "!aa''aa'''aa'a|",
    "L~+'aa'=~+Xa=~J",
    ";_]'aX'[,L~J;_,",
    "!XX'a'''[___]a|",
    "!''a''''XX'''a|",
    "!'r^^^l'X'A'A'|",
    "!'zv#*#j''A''a|",
    "!''<#*#>''A'AX|",
    "!''zv*vj'A'A'&|",
    "L+'''''''''''=J",
    "/L~~~~~~~~~~~~J/"])

MINI_PREDATOR_PREY_FOREST = _make_map([
    "/;____________,/",
    ";]XAXXXXXAXXAX[,",
    "!XXXXXXXXXXXXXX|",
    "!''M'M'MMM'M''|",
    "!'MQ'MMMMM''QM|",
    "!''''QM''MQ'''|",
    "!M'MMMM@@MMMMM|",
    "!M'''@@@@@''''M|",
    "!Q'MQ@@A@@QM'Q|",
    "!M'''@@@@@''''M|",
    "!M'MMMM@@MMMMM|",
    "!''''QM''MQ'''|",
    "!'MQ'MMMMM''QM|",
    "!''M'M'MMM'M''|",
    "!XXXXXXXXXXXXXX|",
    "L+XAXXXXXAXXAX=J"])

# allelopathic_harvest__open: 28x30 -> 16x16 (no walls, just P and berries)
MINI_ALLELOPATHIC_HARVEST = _make_map([
    "3PP12PP322P32PP3",
    "1PPP2P122P3P2312",
    "P1P311PP13PP31PP",
    "PPP2P2P1P2P3P33P",
    "P1PPPPP2PP123111",
    "133P2PP2PP3PP1PP",
    "3PPPPPPPPPP31PPP",
    "PP2P21P21P33PPPP",
    "PPP1P1P32P3PP22P",
    "PPP3P312221PPP22",
    "P121P1PP1P321PPP",
    "PPP22P12PPP1PPPP",
    "PP2PP3P2P1PP1P2P",
    "23PP2PPP2P3PPP3P",
    "2PPP3P3P3P3PP3PP",
    "21PP2PP331PP3PPP"])

# fruit_market: 31x31 -> 16x16 (1 ring instead of 3)
MINI_FRUIT_MARKET = _make_map([
    "xxxxxxxxxxxxxxxx",
    "x/____________+x",
    "x'############`x",
    "x!~~~~~~~~~~~~|x",
    "x!~~~~~~~~~~~~|x",
    "x!~~LLLLLLLL~~|x",
    "x!~~LP~P~P~L~~|x",
    "x!~~L~P~P~~L~~|x",
    "x!~~L~~P~P~L~~|x",
    "x!~~L~P~P~~L~~|x",
    "x!~~LP~P~P~L~~|x",
    "x!~~LLLLLLLL~~|x",
    "x!~~~~~~~~~~~~|x",
    "x!~~~~~~~~~~~~|x",
    "x(------------)x",
    "x<############>x"])


# =============================================================================
# GAME -> MAP REGISTRY
# =============================================================================

MINI_MAPS = {
    # Tier 1
    'daycare': MINI_DAYCARE,
    'externality_mushrooms__dense': MINI_EXTERNALITY_MUSHROOMS_DENSE,

    # Tier 2A: In-the-matrix arena (7 games, same template)
    'bach_or_stravinsky_in_the_matrix__arena': MINI_MATRIX_ARENA,
    'chicken_in_the_matrix__arena': MINI_MATRIX_ARENA,
    'prisoners_dilemma_in_the_matrix__arena': MINI_MATRIX_ARENA,
    'pure_coordination_in_the_matrix__arena': MINI_MATRIX_ARENA,
    'rationalizable_coordination_in_the_matrix__arena': MINI_MATRIX_ARENA,
    'running_with_scissors_in_the_matrix__arena': MINI_MATRIX_ARENA,
    'stag_hunt_in_the_matrix__arena': MINI_MATRIX_ARENA,

    # Tier 2A: In-the-matrix repeated (7 games, same template)
    'bach_or_stravinsky_in_the_matrix__repeated': MINI_MATRIX_REPEATED,
    'chicken_in_the_matrix__repeated': MINI_MATRIX_REPEATED,
    'prisoners_dilemma_in_the_matrix__repeated': MINI_MATRIX_REPEATED,
    'pure_coordination_in_the_matrix__repeated': MINI_MATRIX_REPEATED,
    'rationalizable_coordination_in_the_matrix__repeated': MINI_MATRIX_REPEATED,
    'stag_hunt_in_the_matrix__repeated': MINI_MATRIX_REPEATED,
    'running_with_scissors_in_the_matrix__repeated': MINI_MATRIX_REPEATED,
    'running_with_scissors_in_the_matrix__one_shot': MINI_RWS_ONE_SHOT,

    # Tier 2B: Chemistry (4 games)
    'chemistry__two_metabolic_cycles': MINI_CHEMISTRY_TWO,
    'chemistry__two_metabolic_cycles_with_distractors': MINI_CHEMISTRY_TWO,
    'chemistry__three_metabolic_cycles': MINI_CHEMISTRY_THREE,
    'chemistry__three_metabolic_cycles_with_plentiful_distractors': MINI_CHEMISTRY_THREE,

    # Tier 2C: Commons harvest
    'commons_harvest__open': MINI_COMMONS_HARVEST_OPEN,
    'commons_harvest__closed': MINI_COMMONS_HARVEST_CLOSED,
    'commons_harvest__partnership': MINI_COMMONS_HARVEST_PARTNERSHIP,

    # Tier 2C: Territory
    'territory__rooms': MINI_TERRITORY_ROOMS,
    'territory__inside_out': MINI_TERRITORY_INSIDE_OUT,

    # Tier 2C: Paintball
    'paintball__capture_the_flag': MINI_PAINTBALL_CTF,
    'paintball__king_of_the_hill': MINI_PAINTBALL_KOTH,

    # Tier 2C: Other
    'factory_commons__either_or': MINI_FACTORY_COMMONS,
    'gift_refinements': MINI_GIFT_REFINEMENTS,
    'coop_mining': MINI_COOP_MINING,

    # Tier 3: Hard
    'clean_up': MINI_CLEAN_UP,
    'predator_prey__open': MINI_PREDATOR_PREY_OPEN,
    'predator_prey__alley_hunt': MINI_PREDATOR_PREY_ALLEY,
    'predator_prey__orchard': MINI_PREDATOR_PREY_ORCHARD,
    'predator_prey__random_forest': MINI_PREDATOR_PREY_FOREST,
    'allelopathic_harvest__open': MINI_ALLELOPATHIC_HARVEST,
    'fruit_market__concentric_rivers': MINI_FRUIT_MARKET,
}

# Reduced player counts for games that need fewer players
MINI_PLAYER_COUNTS = {
    # Arena variants: 8 -> 4
    'bach_or_stravinsky_in_the_matrix__arena': 4,
    'chicken_in_the_matrix__arena': 4,
    'prisoners_dilemma_in_the_matrix__arena': 4,
    'pure_coordination_in_the_matrix__arena': 4,
    'rationalizable_coordination_in_the_matrix__arena': 4,
    'running_with_scissors_in_the_matrix__arena': 4,
    'stag_hunt_in_the_matrix__arena': 4,

    # Chemistry: 8 -> 4
    'chemistry__two_metabolic_cycles': 4,
    'chemistry__two_metabolic_cycles_with_distractors': 4,
    'chemistry__three_metabolic_cycles': 4,
    'chemistry__three_metabolic_cycles_with_plentiful_distractors': 4,

    # Commons harvest: 7 -> 4
    'commons_harvest__open': 4,
    'commons_harvest__closed': 4,
    'commons_harvest__partnership': 4,

    # Collaborative cooking crowded: 9 -> 5
    'collaborative_cooking__crowded': 5,

    # Territory rooms: 9 -> 4
    'territory__rooms': 4,

    # Paintball: 8 -> 4
    'paintball__capture_the_flag': 4,
    'paintball__king_of_the_hill': 4,

    # Externality mushrooms: 5 -> 3
    'externality_mushrooms__dense': 3,

    # Gift refinements: 6 -> 4
    'gift_refinements': 4,

    # Coop mining: 6 -> 4
    'coop_mining': 4,

    # Clean up: 7 -> 4
    'clean_up': 4,

    # Predator prey: 13 -> 7 (2 pred + 5 prey)
    'predator_prey__open': 7,
    'predator_prey__alley_hunt': 7,
    'predator_prey__orchard': 7,
    'predator_prey__random_forest': 7,

    # Allelopathic harvest: 16 -> 4
    'allelopathic_harvest__open': 4,

    # Fruit market: 16 -> 4
    'fruit_market__concentric_rivers': 4,
}

# Games that already fit 16x16 without changes
GAMES_ALREADY_FIT = [
    'coins',
    'collaborative_cooking__asymmetric',
    'collaborative_cooking__circuit',
    'collaborative_cooking__cramped',
    'collaborative_cooking__crowded',
    'collaborative_cooking__figure_eight',
    'collaborative_cooking__forced',
    'collaborative_cooking__ring',
]

# All 46 games that can be used at 512x512
MINI_GAMES_512 = sorted(GAMES_ALREADY_FIT + list(MINI_MAPS.keys()))


# =============================================================================
# 256x256 MINI MAPS (8x8 grid, 32px sprites, 4x upscale)
# =============================================================================
# For 256x256 canvas: 256 / 32 = 8 cells per side.
# Native: 8 * 8px = 64x64, scaled 4x -> 256x256.
# Player sprites: 8px native -> 32px output (same as coins dataset).

# --- Tier 1: Already done ---
MINI_ALLELOPATHIC_HARVEST_256 = _make_map([
    "3PP12P23",
    "1PP2P1P2",
    "P1P3PP3P",
    "PP2P1PPP",
    "P1PP3P2P",
    "3PP2PP1P",
    "P2P3P3PP",
    "21P2P3PP"], target_w=8)

# --- Tier 2: Collaborative cooking (trim 1 left x col, pad to 8 rows) ---
# Ring: 9x5 -> 8x5 -> 8x8 (trimmed left x)
MINI_COOKING_RING_256 = _make_map([
    "xxxxxxxx",
    "x###C#xx",
    "x#   Cxx",
    "xDP# #xx",
    "xO P #xx",
    "x#OT##xx",
    "xxxxxxxx",
    "xxxxxxxx"], target_w=8)

# Forced: 9x5 -> 8x5 -> 8x8
MINI_COOKING_FORCED_256 = _make_map([
    "xxxxxxxx",
    "x###C#xx",
    "xO #PCxx",
    "xOP# #xx",
    "xD # #xx",
    "x###T#xx",
    "xxxxxxxx",
    "xxxxxxxx"], target_w=8)

# Cramped: 9x5 -> 8x5 -> 8x8
MINI_COOKING_CRAMPED_256 = _make_map([
    "xxxxxxxx",
    "x##C##xx",
    "xOP POxx",
    "x#   #xx",
    "x#D#T#xx",
    "xxxxxxxx",
    "xxxxxxxx",
    "xxxxxxxx"], target_w=8)

# Circuit: 9x5 -> 8x5 -> 8x8 (trimmed left x)
MINI_COOKING_CIRCUIT_256 = _make_map([
    "xxxxxxxx",
    "###CC###",
    "#P     #",
    "D #### T",
    "#     P#",
    "###OO###",
    "xxxxxxxx",
    "xxxxxxxx"], target_w=8)

# Asymmetric: 9x5 -> redesigned to 8x5 -> 8x8
MINI_COOKING_ASYMMETRIC_256 = _make_map([
    "xxxxxxxx",
    "########",
    "O #T#O T",
    "#P C  P#",
    "#  C   #",
    "###DD###",
    "xxxxxxxx",
    "xxxxxxxx"], target_w=8)

# --- Tier 3: Open grid games ---
# Externality mushrooms dense: wall border + R/G/B/O mushrooms
# space = dirt + spawn_point + potential_mushroom
MINI_EXTERNALITY_MUSHROOMS_256 = _make_map([
    "/______+",
    "! R  G |",
    "!  R   |",
    "! B  O |",
    "!   B  |",
    "! G  R |",
    "!  O   |",
    "(------)"], target_w=8)

# Daycare: wall border + P spawn + ~ trees/fruit
MINI_DAYCARE_256 = _make_map([
    "/______+",
    "!~~  ~~|",
    "!~ PP ~|",
    "!~ ~~ ~|",
    "!~ ~~ ~|",
    "!~ PP ~|",
    "!~~  ~~|",
    "(------)"], target_w=8)

# Commons harvest open: W border + P/Q spawn + A apples (clustered for regrowth)
# Q = inside_spawn_point (required for 'insideSpawnPoints' group)
MINI_COMMONS_HARVEST_OPEN_256 = _make_map([
    "WWWWWWWW",
    "WP A QPW",
    "W AAAA W",
    "W A AA W",
    "W AA A W",
    "W AAAA W",
    "WQA  QPW",
    "WWWWWWWW"], target_w=8)

# --- Tier 4: Matrix games (shared templates) ---
# Arena: W border, P spawns, 1/2/a resources
MINI_MATRIX_ARENA_256 = _make_map([
    "WWWWWWWW",
    "WP 12aPW",
    "W 1aa2 W",
    "W  aa  W",
    "W  aa  W",
    "W 2aa1 W",
    "WPa21 PW",
    "WWWWWWWW"], target_w=8)

# Repeated: W border, n spawns, 1/2/a resources
MINI_MATRIX_REPEATED_256 = _make_map([
    "WWWWWWWW",
    "Wn12a nW",
    "W 1aa2 W",
    "W  aa  W",
    "W  aa  W",
    "W 2aa1 W",
    "Wn a21nW",
    "WWWWWWWW"], target_w=8)

# Running with scissors one-shot: W border, n/r/s/a/p resources
MINI_RWS_ONE_SHOT_256 = _make_map([
    "WWWWWWWW",
    "Wn ra nW",
    "W r aa W",
    "W  aa  W",
    "W  aa  W",
    "W sa p W",
    "Wn ps nW",
    "WWWWWWWW"], target_w=8)

# --- Tier 5: Harder games ---
# Commons harvest closed: W border + wall dividers + A apples + Q spawns
MINI_COMMONS_HARVEST_CLOSED_256 = _make_map([
    "WWWWWWWW",
    "WP AAQPW",
    "W AAA  W",
    "WW   WWW",
    "WWW   WW",
    "W  AAA W",
    "WPAA QPW",
    "WWWWWWWW"], target_w=8)

# Commons harvest partnership: W border + I dividers + A apples + Q spawns
MINI_COMMONS_HARVEST_PARTNERSHIP_256 = _make_map([
    "WWWWWWWW",
    "WPAA QPW",
    "W AA I W",
    "W    I W",
    "W I    W",
    "W I AA W",
    "WQA AAPW",
    "WWWWWWWW"], target_w=8)

# Coop mining: W border + O ore
MINI_COOP_MINING_256 = _make_map([
    "WWWWWWWW",
    "WP OO PW",
    "W  OO  W",
    "W      W",
    "W      W",
    "W  OO  W",
    "WP OO PW",
    "WWWWWWWW"], target_w=8)

# Gift refinements: W border + T tokens
MINI_GIFT_REFINEMENTS_256 = _make_map([
    "WWWWWWWW",
    "WP TT PW",
    "W  TT  W",
    "W      W",
    "W      W",
    "W  TT  W",
    "WP TT PW",
    "WWWWWWWW"], target_w=8)

# Predator prey open: wall border, * prey spawn, X predator spawn, a apple, & apple+prey spawn
MINI_PREDATOR_PREY_OPEN_256 = _make_map([
    "/;____,/",
    ";]****[,",
    "!'Xaa&'|",
    "!''aa''|",
    "!'&aaX'|",
    "!''aa''|",
    "L+****=J",
    "/L~~~~J/"], target_w=8)

# =============================================================================
# 256x256 REGISTRIES
# =============================================================================

MINI_MAPS_256 = {
    # Tier 1: Already done
    'allelopathic_harvest__open': MINI_ALLELOPATHIC_HARVEST_256,

    # Tier 2: Collaborative cooking
    'collaborative_cooking__ring': MINI_COOKING_RING_256,
    'collaborative_cooking__forced': MINI_COOKING_FORCED_256,
    'collaborative_cooking__cramped': MINI_COOKING_CRAMPED_256,
    'collaborative_cooking__circuit': MINI_COOKING_CIRCUIT_256,
    'collaborative_cooking__asymmetric': MINI_COOKING_ASYMMETRIC_256,

    # Tier 3: Open grid games
    'externality_mushrooms__dense': MINI_EXTERNALITY_MUSHROOMS_256,
    'daycare': MINI_DAYCARE_256,
    'commons_harvest__open': MINI_COMMONS_HARVEST_OPEN_256,

    # Tier 4: Matrix arena (7 games, shared template)
    'bach_or_stravinsky_in_the_matrix__arena': MINI_MATRIX_ARENA_256,
    'chicken_in_the_matrix__arena': MINI_MATRIX_ARENA_256,
    'prisoners_dilemma_in_the_matrix__arena': MINI_MATRIX_ARENA_256,
    'pure_coordination_in_the_matrix__arena': MINI_MATRIX_ARENA_256,
    'rationalizable_coordination_in_the_matrix__arena': MINI_MATRIX_ARENA_256,
    'running_with_scissors_in_the_matrix__arena': MINI_MATRIX_ARENA_256,
    'stag_hunt_in_the_matrix__arena': MINI_MATRIX_ARENA_256,

    # Tier 4: Matrix repeated (7 games, shared template)
    'bach_or_stravinsky_in_the_matrix__repeated': MINI_MATRIX_REPEATED_256,
    'chicken_in_the_matrix__repeated': MINI_MATRIX_REPEATED_256,
    'prisoners_dilemma_in_the_matrix__repeated': MINI_MATRIX_REPEATED_256,
    'pure_coordination_in_the_matrix__repeated': MINI_MATRIX_REPEATED_256,
    'rationalizable_coordination_in_the_matrix__repeated': MINI_MATRIX_REPEATED_256,
    'stag_hunt_in_the_matrix__repeated': MINI_MATRIX_REPEATED_256,
    'running_with_scissors_in_the_matrix__repeated': MINI_MATRIX_REPEATED_256,

    # Tier 4: RWS one-shot
    'running_with_scissors_in_the_matrix__one_shot': MINI_RWS_ONE_SHOT_256,

    # Tier 5: Harder games
    'commons_harvest__closed': MINI_COMMONS_HARVEST_CLOSED_256,
    'commons_harvest__partnership': MINI_COMMONS_HARVEST_PARTNERSHIP_256,
    'coop_mining': MINI_COOP_MINING_256,
    'gift_refinements': MINI_GIFT_REFINEMENTS_256,
    'predator_prey__open': MINI_PREDATOR_PREY_OPEN_256,
}

MINI_PLAYER_COUNTS_256 = {
    # Tier 1
    'allelopathic_harvest__open': 2,

    # Tier 2: Cooking games (use native 2 players)
    # collaborative_cooking games default to 2 players, no override needed

    # Tier 3: Open grid games -> 2 players
    'externality_mushrooms__dense': 2,
    'daycare': 2,
    'commons_harvest__open': 2,

    # Tier 4: Matrix arena -> 2 players (was 8)
    'bach_or_stravinsky_in_the_matrix__arena': 2,
    'chicken_in_the_matrix__arena': 2,
    'prisoners_dilemma_in_the_matrix__arena': 2,
    'pure_coordination_in_the_matrix__arena': 2,
    'rationalizable_coordination_in_the_matrix__arena': 2,
    'running_with_scissors_in_the_matrix__arena': 2,
    'stag_hunt_in_the_matrix__arena': 2,

    # Tier 4: Matrix repeated -> 2 players (already 2 for most)
    'bach_or_stravinsky_in_the_matrix__repeated': 2,
    'chicken_in_the_matrix__repeated': 2,
    'prisoners_dilemma_in_the_matrix__repeated': 2,
    'pure_coordination_in_the_matrix__repeated': 2,
    'rationalizable_coordination_in_the_matrix__repeated': 2,
    'stag_hunt_in_the_matrix__repeated': 2,
    'running_with_scissors_in_the_matrix__repeated': 2,
    'running_with_scissors_in_the_matrix__one_shot': 2,

    # Tier 5: Harder games -> 2-3 players
    'commons_harvest__closed': 2,
    'commons_harvest__partnership': 2,
    'coop_mining': 2,
    'gift_refinements': 2,
    'predator_prey__open': 3,  # 1 predator + 2 prey
}

# Explicit role selections for games where simple truncation doesn't work
# (e.g., predator_prey needs a mix of predator and prey roles)
MINI_ROLES_256 = {
    'predator_prey__open': ('predator', 'prey', 'prey'),
}

MINI_GAMES_256 = sorted(list(MINI_MAPS_256.keys()))


# =============================================================================
# VALIDATION MAPS (dense spawn points for position diversity)
# =============================================================================

def _fill_spawns(map_str, walkable_chars, spawn_char):
    """Replace walkable cells with spawn points for maximum position diversity.

    Only replaces characters in walkable_chars with spawn_char.
    Walls, resources, and other functional tiles are preserved.
    """
    lines = map_str.split('\n')
    result = []
    for line in lines:
        new_line = []
        for ch in line:
            if ch in walkable_chars:
                new_line.append(spawn_char)
            else:
                new_line.append(ch)
        result.append(''.join(new_line))
    return '\n'.join(result)


# Matrix arena: ' ' -> 'P' (floor -> floor+spawn_point)
MINI_MATRIX_ARENA_VAL = _fill_spawns(MINI_MATRIX_ARENA, ' ', 'P')

# Matrix repeated: ' ' -> 'n' (floor -> floor+spawn_point)
MINI_MATRIX_REPEATED_VAL = _fill_spawns(MINI_MATRIX_REPEATED, ' ', 'n')

# RWS one-shot: ' ' -> 'n'
MINI_RWS_ONE_SHOT_VAL = _fill_spawns(MINI_RWS_ONE_SHOT, ' ', 'n')

# Commons harvest: ' ' -> 'P' (floor -> floor+spawn_point)
MINI_COMMONS_HARVEST_OPEN_VAL = _fill_spawns(MINI_COMMONS_HARVEST_OPEN, ' ', 'P')
MINI_COMMONS_HARVEST_CLOSED_VAL = _fill_spawns(MINI_COMMONS_HARVEST_CLOSED, ' ', 'P')
MINI_COMMONS_HARVEST_PARTNERSHIP_VAL = _fill_spawns(MINI_COMMONS_HARVEST_PARTNERSHIP, ' ', 'P')

# Territory: ',' -> 'P' (floor -> floor+spawn_point)
MINI_TERRITORY_ROOMS_VAL = _fill_spawns(MINI_TERRITORY_ROOMS, ',', 'P')
MINI_TERRITORY_INSIDE_OUT_VAL = _fill_spawns(MINI_TERRITORY_INSIDE_OUT, ',', 'P')

# Clean up: ' ' -> 'P' (sand -> sand+spawn_point)
MINI_CLEAN_UP_VAL = _fill_spawns(MINI_CLEAN_UP, ' ', 'P')

MINI_MAPS_VAL = {
    # Matrix arena (7 games)
    'bach_or_stravinsky_in_the_matrix__arena': MINI_MATRIX_ARENA_VAL,
    'chicken_in_the_matrix__arena': MINI_MATRIX_ARENA_VAL,
    'prisoners_dilemma_in_the_matrix__arena': MINI_MATRIX_ARENA_VAL,
    'pure_coordination_in_the_matrix__arena': MINI_MATRIX_ARENA_VAL,
    'rationalizable_coordination_in_the_matrix__arena': MINI_MATRIX_ARENA_VAL,
    'running_with_scissors_in_the_matrix__arena': MINI_MATRIX_ARENA_VAL,
    'stag_hunt_in_the_matrix__arena': MINI_MATRIX_ARENA_VAL,

    # Matrix repeated (7 games)
    'bach_or_stravinsky_in_the_matrix__repeated': MINI_MATRIX_REPEATED_VAL,
    'chicken_in_the_matrix__repeated': MINI_MATRIX_REPEATED_VAL,
    'prisoners_dilemma_in_the_matrix__repeated': MINI_MATRIX_REPEATED_VAL,
    'pure_coordination_in_the_matrix__repeated': MINI_MATRIX_REPEATED_VAL,
    'rationalizable_coordination_in_the_matrix__repeated': MINI_MATRIX_REPEATED_VAL,
    'stag_hunt_in_the_matrix__repeated': MINI_MATRIX_REPEATED_VAL,
    'running_with_scissors_in_the_matrix__repeated': MINI_MATRIX_REPEATED_VAL,
    'running_with_scissors_in_the_matrix__one_shot': MINI_RWS_ONE_SHOT_VAL,

    # Commons harvest (3 games)
    'commons_harvest__open': MINI_COMMONS_HARVEST_OPEN_VAL,
    'commons_harvest__closed': MINI_COMMONS_HARVEST_CLOSED_VAL,
    'commons_harvest__partnership': MINI_COMMONS_HARVEST_PARTNERSHIP_VAL,

    # Territory (2 games)
    'territory__rooms': MINI_TERRITORY_ROOMS_VAL,
    'territory__inside_out': MINI_TERRITORY_INSIDE_OUT_VAL,

    # Clean up (1 game)
    'clean_up': MINI_CLEAN_UP_VAL,
}


# =============================================================================
# CONFIG APPLICATION
# =============================================================================

def get_mini_map(game_name):
    """Get the mini ASCII map for a game, or None if it already fits."""
    return MINI_MAPS.get(game_name, None)


def get_mini_player_count(game_name):
    """Get the reduced player count for a game, or None to use default."""
    return MINI_PLAYER_COUNTS.get(game_name, None)


def apply_mini_config_val(game_name, config):
    """Apply mini map with dense spawn points for validation.

    Uses MINI_MAPS_VAL (which fill walkable cells with spawn markers) when
    available, otherwise falls back to the standard MINI_MAPS. This gives
    much more diverse initial player positions for validation episodes.
    """
    if game_name == 'coins':
        config.min_width = 14
        config.max_width = 14
        config.min_height = 14
        config.max_height = 14
        return config, config.default_player_roles

    mini_count = get_mini_player_count(game_name)
    if mini_count is not None:
        original_roles = config.default_player_roles
        if mini_count < len(original_roles):
            config.default_player_roles = original_roles[:mini_count]
        roles = config.default_player_roles
    else:
        roles = config.default_player_roles

    val_map = MINI_MAPS_VAL.get(game_name, None)
    mini_map = val_map if val_map is not None else get_mini_map(game_name)
    if mini_map is not None:
        original_builder = config.lab2d_settings_builder

        def _wrapped_builder(*, config, roles):
            settings = original_builder(config=config, roles=roles)
            if 'simulation' in settings and 'map' in settings['simulation']:
                settings['simulation']['map'] = mini_map
            else:
                print(f"  WARNING: Could not find simulation.map for {game_name}")
            return settings

        config.lab2d_settings_builder = _wrapped_builder

    return config, roles


def apply_mini_config(game_name, config):
    """Apply mini map and player count to a substrate config.

    Call this after get_config() but before build_from_config().
    Wraps the lab2d_settings_builder to inject the mini map into the
    generated settings['simulation']['map'].

    Returns:
        config: modified config
        roles: tuple of player roles to use
    """
    # Special case: coins has procedural maps
    if game_name == 'coins':
        config.min_width = 14
        config.max_width = 14
        config.min_height = 14
        config.max_height = 14
        return config, config.default_player_roles

    # Apply reduced player count first
    mini_count = get_mini_player_count(game_name)
    if mini_count is not None:
        original_roles = config.default_player_roles
        if mini_count < len(original_roles):
            config.default_player_roles = original_roles[:mini_count]
        roles = config.default_player_roles
    else:
        roles = config.default_player_roles

    # Apply mini map by wrapping the lab2d_settings_builder
    mini_map = get_mini_map(game_name)
    if mini_map is not None:
        original_builder = config.lab2d_settings_builder

        def _wrapped_builder(*, config, roles):
            settings = original_builder(config=config, roles=roles)
            if 'simulation' in settings and 'map' in settings['simulation']:
                settings['simulation']['map'] = mini_map
            else:
                print(f"  WARNING: Could not find simulation.map for {game_name}")
            return settings

        config.lab2d_settings_builder = _wrapped_builder

    return config, roles


def apply_mini_config_256(game_name, config):
    """Apply 256x256 mini map (8x8 grid) and player count to a substrate config.

    Same as apply_mini_config but uses the 256-resolution maps (8x8 grid).
    These give 32x32 pixel player sprites at 256x256 output (4x upscale).

    Handles two map injection strategies:
    1. For games with config.layout.ascii_map (e.g. cooking): modify layout
       directly so the builder generates correct prefab placement.
    2. For all other games: wrap the builder to override settings.simulation.map.

    Returns:
        config: modified config
        roles: tuple of player roles to use
    """
    # Apply explicit role selection if available, else reduce player count
    explicit_roles = MINI_ROLES_256.get(game_name, None)
    if explicit_roles is not None:
        config.default_player_roles = explicit_roles
        roles = explicit_roles
    else:
        mini_count = MINI_PLAYER_COUNTS_256.get(game_name, None)
        if mini_count is not None:
            original_roles = config.default_player_roles
            if mini_count < len(original_roles):
                config.default_player_roles = original_roles[:mini_count]
        roles = config.default_player_roles

    # Apply mini map
    mini_map = MINI_MAPS_256.get(game_name, None)
    if mini_map is not None:
        # Strategy 1: If config has layout.ascii_map, modify it directly.
        # This is critical for games like collaborative_cooking where the
        # builder uses ascii_map to set up prefab components (Container etc.)
        # For these games, we must NOT override simulation.map afterwards
        # because the builder generates additional Lua state from the layout.
        has_layout = (hasattr(config, 'layout') and
                      hasattr(config.layout, 'ascii_map'))
        if has_layout:
            # Maps set via layout.ascii_map must start with \n because
            # get_game_object_positions_from_map does rows[1:] to skip
            # the empty first element from splitting a leading-\n string.
            # Without this, Python object positions are off by 1 row
            # vs Lua rendering positions.
            if not mini_map.startswith('\n'):
                mini_map_for_layout = '\n' + mini_map
            else:
                mini_map_for_layout = mini_map
            config.layout.ascii_map = mini_map_for_layout
        else:
            # Strategy 2: Wrap the builder to override simulation.map
            # (for games without layout.ascii_map)
            original_builder = config.lab2d_settings_builder

            def _wrapped_builder(*, config, roles):
                settings = original_builder(config=config, roles=roles)
                if 'simulation' in settings and 'map' in settings['simulation']:
                    settings['simulation']['map'] = mini_map
                else:
                    print(f"  WARNING: Could not find simulation.map for {game_name}")
                return settings

            config.lab2d_settings_builder = _wrapped_builder

    return config, roles


# =============================================================================
# VALIDATION
# =============================================================================

def validate_all_maps(max_size=16):
    """Validate that all mini maps fit within the target grid size."""
    errors = []
    for game_name, ascii_map in sorted(MINI_MAPS.items()):
        lines = ascii_map.strip().split('\n')
        height = len(lines)
        widths = [len(line) for line in lines]
        width = max(widths) if widths else 0
        min_width = min(widths) if widths else 0

        # territory__rooms uses TORUS and can be slightly wider
        game_max = 17 if 'territory__rooms' in game_name else max_size

        if width > game_max:
            errors.append(f"{game_name}: width {width} > {game_max}")
        if height > max_size:
            errors.append(f"{game_name}: height {height} > {max_size}")
        if min_width != width:
            errors.append(f"{game_name}: inconsistent widths ({min_width}-{width})")

    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  {e}")
        return False
    else:
        print(f"All {len(MINI_MAPS)} maps validated OK (max {max_size}x{max_size})")
        # Print dimensions
        for game_name, ascii_map in sorted(MINI_MAPS.items()):
            lines = ascii_map.strip().split('\n')
            w = len(lines[0])
            h = len(lines)
            print(f"  {game_name}: {w}x{h}")
        return True


if __name__ == '__main__':
    ok = validate_all_maps()
    print(f"\nTotal games in MINI_GAMES_512: {len(MINI_GAMES_512)}")
    print(f"  Already fit: {len(GAMES_ALREADY_FIT)}")
    print(f"  Need mini maps: {len(MINI_MAPS)}")
    print(f"  Need reduced players: {len(MINI_PLAYER_COUNTS)}")
    if not ok:
        exit(1)
