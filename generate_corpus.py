import os
from settings import get_settings

def generate_math_science_corpus():
    corpus_path = get_settings().raw_data_dir / "corpus.txt"
    print(f"[*] Generating math, logic, and science corpus at {corpus_path}...")
    
    sections = [
        "Mathematics is the abstract study of numbers, quantity, structure, space, and change. Arithmetic operations include addition, subtraction, multiplication, and division.",
        "Algebra involves variables and equations. For example, if 2x + 3 = 7, then 2x = 4, so x = 2. Quadratic equations take the form ax^2 + bx + c = 0.",
        "Calculus studies continuous change through derivatives and integrals. The derivative of x^2 with respect to x is 2x. The integral of 2x dx is x^2 + C.",
        "Geometry examines shapes, angles, dimensions, and properties of space. The Pythagorean theorem states that a^2 + b^2 = c^2 in a right-angled triangle.",
        "Logic and reasoning form the foundation of formal proofs. If statement P implies Q, and P is true, then Q must be true by modus ponens.",
        "Python programming fundamentals: def add(a, b): return a + b computes the sum of two numbers. Lists, dictionaries, and loops enable data manipulation.",
        "Physics and thermodynamics: Newton's second law states that Force equals mass times acceleration (F = ma). Energy is conserved in isolated systems.",
        "Probability and statistics analyze uncertainty. The probability of an event is the number of favorable outcomes divided by total outcomes."
    ]
    
    corpus_text = "\n\n".join(sections * 15)

    os.makedirs(corpus_path.parent, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(corpus_text)
        
    print(f"[+] Math and science corpus generated successfully! ({len(corpus_text)} characters)")

if __name__ == "__main__":
    generate_math_science_corpus()
