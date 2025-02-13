type_map = {0: "kid", 1: "youth", 2: "student", 3: "adult", 4: "old"}
rvrs_map = {"kid": 0, "youth": 1, "student": 2, "adult": 3, "old": 4}

class Fixed_Household:
    __slots__ = (
        "area",
        "id",
        "composition",
        "num_people",
        "total_size",
    )

    def __init__(self, area, id, composition, size):
        self.area = area
        self.id = id
        self.composition = composition
        self.total_size = size

        self.num_people = {category: composition[i] for i, category in type_map.items()}

    def __repr__(self):
        num_people = [self.num_people[category] for category in type_map.values()]
        return f"id:{self.id}, total_size:{self.total_size}, People: {num_people}"
