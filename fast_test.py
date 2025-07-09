from source.train.generator import MusicGenerator

gen = MusicGenerator("models/checkpoints/best_model.pt", device="cpu")

result1 = gen.generate_music(
    "A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle.",
    "o1.mid",
)
result2 = gen.generate_music(
    "Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach",
    "o2.mid",
)
print(result1["generated_tokens"][:40])
print(result2["generated_tokens"][:40])
