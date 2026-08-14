import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

def run_smol_sft():
    print("[*] Starting Supervised Fine-Tuning on SmolLM2-135M Base Model...")
    base_model_id = "HuggingFaceTB/SmolLM2-135M"
    output_dir = "C:\\Users\\Nyxentra\\Desktop\\local_llm\\output_model"

    print(f"[*] Downloading / loading base model: {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 50+ Diverse Q&A Instruction Pairs
    instructions = [
        "Hi", "Hello", "Hey", "What is Artificial Intelligence?", "What is Python?",
        "Solve 2x + 3 = 7", "What is the derivative of x^2?", "Explain the Pythagorean theorem",
        "How do I write a function in Python?", "What is a Transformer?", "Why use local LLMs?",
        "What is machine learning?", "What is calculus?", "Solve x + 5 = 12", "What is 15 * 4?",
        "What is an array?", "What is a neural network?", "What is data science?", "How does gradient descent work?",
        "What is a variable in programming?", "What is HTML?", "What is CSS?", "What is JavaScript?",
        "What is an API?", "What is cloud computing?", "What is cybersecurity?", "What is an algorithm?",
        "What is recursion?", "What is object-oriented programming?", "What is a database?", "What is SQL?",
        "What is Linux?", "What is Docker?", "What is Git?", "What is GitHub?", "What is open source?",
        "What is software engineering?", "What is debugging?", "What is unit testing?", "What is agile methodology?",
        "What is CI/CD?", "What is microservices?", "What is REST API?", "What is JSON?", "What is XML?",
        "What is HTTP?", "What is TCP/IP?", "What is DNS?", "What is encryption?", "What is a hash function?"
    ]

    responses = [
        "Hello! I am your local AI assistant powered by SmolLM2. How can I help you today?",
        "Hello! I am your local AI assistant powered by SmolLM2. How can I help you today?",
        "Hello! I am your local AI assistant powered by SmolLM2. How can I help you today?",
        "Artificial Intelligence is the field of creating computer systems that can perform tasks that normally require human intelligence, such as learning, reasoning and problem solving.",
        "Python is a high-level, interpreted programming language known for its simple syntax and readability. It is widely used in AI, data science and web development.",
        "To solve 2x + 3 = 7, subtract 3 from both sides to get 2x = 4, then divide by 2 to get x = 2.",
        "The derivative of x^2 is 2x, using the power rule in calculus.",
        "The Pythagorean theorem states that in a right triangle, a² + b² = c² where c is the hypotenuse.",
        "You define a function in Python with the def keyword, for example:\ndef greet(name):\n    return f'Hello, {name}!'",
        "A Transformer is a neural network architecture that uses self-attention to process sequences of data efficiently.",
        "Local LLMs give you privacy, offline use, zero API cost and full control over the model.",
        "Machine learning is a subset of AI that allows systems to learn patterns from data without being explicitly programmed.",
        "Calculus is the mathematical study of continuous change, consisting of differential and integral calculus.",
        "Subtract 5 from both sides: x = 7.",
        "15 multiplied by 4 is 60.",
        "An array is a data structure consisting of a collection of elements identified by index or key.",
        "A neural network is a network of artificial neurons inspired by the human brain, used in deep learning.",
        "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge from data.",
        "Gradient descent is an optimization algorithm used to minimize loss by iteratively moving in the direction of steepest descent.",
        "A variable is a storage location paired with an associated symbolic name, which contains some known or unknown quantity of information.",
        "HTML is the standard markup language for documents designed to be displayed in a web browser.",
        "CSS is a style sheet language used for describing the presentation of a document written in a markup language.",
        "JavaScript is a programming language that conforms to the ECMAScript specification and is a core technology of the web.",
        "An API is a set of subroutine definitions, protocols, and tools for building application software.",
        "Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power.",
        "Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks.",
        "An algorithm is a finite sequence of rigorous instructions, typically used to solve a class of specific problems or to perform a computation.",
        "Recursion is a method of solving a problem where the solution depends on solutions to smaller instances of the same problem.",
        "Object-oriented programming is a programming paradigm based on the concept of objects, which can contain data and code.",
        "A database is an organized collection of data or a type of data stored electronically in a computer system.",
        "SQL is a domain-specific language used in programming and designed for managing data held in a relational database management system.",
        "Linux is a family of open-source Unix-like operating systems based on the Linux kernel.",
        "Docker is a set of platform-as-a-service products that use OS-level virtualization to deliver software in packages called containers.",
        "Git is a distributed version control system for tracking changes in source code during software development.",
        "GitHub is a web-based hosting service for version control using Git.",
        "Open source denotes software that permits users to use, change, and improve its source code and to redistribute it in modified or unmodified forms.",
        "Software engineering is a systematic engineering approach to software development.",
        "Debugging is the process of identifying and removing errors from computer hardware or software code.",
        "Unit testing is a software testing method by which individual units of source code are tested to determine whether they are fit for use.",
        "Agile methodology is an iterative approach to project management and software development that helps teams deliver value to their customers faster.",
        "CI/CD is a method to frequently deliver apps to customers by introducing automation into the stages of app development.",
        "Microservices is an architectural style that structures an application as a collection of services.",
        "REST API is an architectural style that defines a set of constraints to be used for creating web services.",
        "JSON is an open standard file format and data interchange format that uses human-readable text to store and transmit data objects.",
        "XML is a markup language that defines a set of rules for encoding documents in a format that is both human-readable and machine-readable.",
        "HTTP is an application layer protocol for distributed, collaborative, hypermedia information systems.",
        "TCP/IP is a set of communication protocols used in the Internet and similar computer networks.",
        "DNS is the hierarchical and decentralized naming system for computers, services, or other resources connected to the Internet.",
        "Encryption is the process of encoding information in such a way that only authorized parties can access it.",
        "A hash function is any function that can be used to map data of arbitrary size to fixed-size values."
    ]

    dataset = Dataset.from_dict({"instruction": instructions, "response": responses})

    def tokenize_function(examples):
        prompts = [f"User: {inst}\nAssistant: {resp}" for inst, resp in zip(examples["instruction"], examples["response"])]
        tokenized = tokenizer(prompts, truncation=True, max_length=128, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "smol_sft_output"),
        per_device_train_batch_size=4,
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("[*] Fine-tuning SmolLM2-135M on expert SFT dataset...")
    trainer.train()

    print(f"[+] Fine-tuning complete! Saving fluent model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    run_smol_sft()
