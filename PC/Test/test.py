class Dog:
    def __init__(self, name, breed="Mixed", age=1):
        self.name = name
        self.breed = breed
        self.age = age

d1 = Dog("Buddy")
d2 = Dog("Max", "Golden Retriever", 5)
print(d1.name, d1.breed, d1.age)  # Вывод: Buddy Mixed 1
print(d2.name, d2.breed, d2.age)  # Вывод: Max Golden Retriever 5
