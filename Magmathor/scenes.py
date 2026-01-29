TEST_HOLDOUT_PLANS = [
    "FloorPlan29",
    "FloorPlan30",
    "FloorPlan229",
    "FloorPlan230",
    "FloorPlan329",
    "FloorPlan330",
    "FloorPlan429",
    "FloorPlan430",
]


class Scenes:

    _kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
    # Remove bad kitchen with object falling through the floor
    _kitchens.remove("FloorPlan8")
    _kitchens.remove("FloorPlan9")
    _kitchens.remove("FloorPlan17")  # Sink is invisible
    _living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
    _bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
    _bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]
    all = _kitchens + _living_rooms + _bedrooms + _bathrooms

    @classmethod
    def get_kitchens(cls, test_set=False):
        if test_set:
            return [k for k in cls._kitchens if k in TEST_HOLDOUT_PLANS]
        else:
            return [k for k in cls._kitchens if k not in TEST_HOLDOUT_PLANS]

    @classmethod
    def get_living_rooms(cls, test_set=False):
        if test_set:
            return [l for l in cls._living_rooms if l in TEST_HOLDOUT_PLANS]
        else:
            return [l for l in cls._living_rooms if l not in TEST_HOLDOUT_PLANS]

    @classmethod
    def get_bedrooms(cls, test_set=False):
        if test_set:
            return [b for b in cls._bedrooms if b in TEST_HOLDOUT_PLANS]
        else:
            return [b for b in cls._bedrooms if b not in TEST_HOLDOUT_PLANS]

    @classmethod
    def get_bathrooms(cls, test_set=False):
        if test_set:
            return [b for b in cls._bathrooms if b in TEST_HOLDOUT_PLANS]
        else:
            return [b for b in cls._bathrooms if b not in TEST_HOLDOUT_PLANS]
