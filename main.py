from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM

def main():
    model_name = "microsoft/biogpt"

    print("Loading model...")
    model = BioGptForCausalLM.from_pretrained(model_name)
    tokenizer = BioGptTokenizer.from_pretrained(model_name)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    set_seed(42)

    prompt = "COVID-19 is"
    results = generator(
        prompt,
        max_length=20,
        num_return_sequences=5,
        do_sample=True
    )

    print("\n=== Generated Text ===\n")
    for i, output in enumerate(results, 1):
        print(f"{i}. {output['generated_text']}")

if __name__ == "__main__":
    main()
