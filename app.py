from tasks import evaluate_all_tasks
import random
import numpy as np

# ensure reproducibility
random.seed(42)
np.random.seed(42)

def main():
    results = evaluate_all_tasks()

    print("Hugging Face Demo Output:")
    for task, score in results.items():
        print(f"{task}: {score:.3f}")

if __name__ == "__main__":
    main()