# Fot US Commodity Market
us_roots = {
    # Metals
    "Metal": ["GC", "SI", "HG", "ALI"],

    # Energy
    "Energy": ["CL", "HO", "RB", "NG"],

    # Agriculture
    "Agriculture": ["ZS", "ZM", "ZL", "ZC", "ZW", "KE", "ZR", "HE",]
}

china_roots = {
    # Metals
    "Metal": ["AU", "AG",              # Precious metals
    "CU", "AL", "ZN", "PB", "NI", "SN", # Base metals                           
    "PT", "PD"],                      # GFEX: shorter-history extension

    # Energy
    "Energy": ["SC", "FU", "LU", "BU"],  # Petroleum sector

    # Agriculture
    "Agriculture": 
    ["B", "M", "Y",                   # Soybean complex
    "C",                              # Corn
    "PM", "WH",                       # Wheat candidates
    "JR", "RI",                       # Rice candidates
    "LH", ]                           # Live hogs
}