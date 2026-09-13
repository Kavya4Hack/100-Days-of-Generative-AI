# Even odd function
# def evenOdd(x):
#     if x%2==0:
#         return "Even Number"
#     else:
#         return "Odd Number"
# x = int(input("Enter Number: "))
# print(evenOdd(x))

# Using *args and **kwargs
"""
*args = It collects extra positional arguments as a tuple.
**kwargs = It collects keyword arguments as a dictionary.
"""
# def myFun(*args, **kwargs):
#     print("Non-Keyword Arguments (*args): ")
#     for arg in args:
#         print(arg)

#     print("Keyword Arguments (**kwargs): ")
#     for key, value in kwargs.items():
#         print(f"{key} == {value}")
# myFun("Hey", "WElcome", first="Kavya", mid="Daxesh", last="Nayi")

# Use of lambda function

# a = "Kavya Nayi"
# upper = lambda x: x.upper()
# print(upper(a))

# x = lambda a : a + 10
# print(x(10))

# Lambda function can take any number of argyments
# x = lambda a, b: a + b * 5
# print(x(2,5)) 

# x = lambda a, b, c : a + b + c
# print(x(10,20,30))

# def myFun(n):
#     return lambda a : a * n
# myRes = myFun(10)
# print(myRes(20))

#  Using lambda with map()
# number = [1,2,3,4,5]
# double = list(map(lambda x : x * 2, number))
# print(double)

#  Use lambda function with filter()
# number = [1,2,3,4,5,6,7,8,9]
# odd_num = list(filter(lambda x : x % 2 !=0, number))
# print(odd_num)

# Use lambda with sorted()
# student = [("Emil", 25), ("Kavya", 21), ("Rushil", 20)]
# sorted = list(sorted(student, key=lambda x : x[1]))
# print(sorted)


# class Student:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

#     def introduce(self):
#         print(f"My name is {self.name} and I am {self.age} years old.")
# Student1 = Student("Kavya", 20)
# Student1.introduce()

# Objects
# class Student:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#     def introduce(self):
#         print(f"My name is {self.name} and I am {self.age} years old.")

# Student1 = Student("Kavya", 20)
# Student2 = Student("Rushil", 20)
# Student3 = Student("Preet", 1)
# Student1.introduce()    #object
# Student2.introduce()    #object
# Student3.introduce()    #object

# Day 2 Project
from abc import ABC, abstractmethod
class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt):
        pass

class GeminiClient(LLMClient):
    def generate(self, prompt):
        return f"Gemini response for: {prompt}"
class OpenAIClient(LLMClient):
    def generate(self, prompt):
        return f"OpenAI response for : {prompt}"

gemini = GeminiClient()
openai = OpenAIClient()

print(gemini.generate("Explain transformers"))
print(openai.generate("Explain transformers"))

def ask_model(client: LLMClient, prompt: str):
    return client.generate(prompt)

gemini = GeminiClient()
openai = OpenAIClient()
print(ask_model(gemini, "What is RAG?"))
print(ask_model(openai, "What is RAG?"))