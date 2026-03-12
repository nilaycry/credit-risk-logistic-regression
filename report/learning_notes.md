# Quant ML & C++ Learning Notes

*This document will continuously track the mathematical concepts, C++ idioms, and quantitative modeling best practices we discuss throughout the project. Use this as a review guide for interviews.*

---

## 1. Machine Learning Data Conventions (Matrices)

### Structuring Dataset Matrices ($X$)
In almost all machine learning contexts (scikit-learn, PyTorch, formal mathematics), tabular data is structured as an $m \times n$ matrix:
*   **Rows ($m$ or $i$) = Samples / Observations.** (e.g., Apple, Ford, Delta Airlines).
*   **Columns ($n$ or $j$) = Features / Variables.** (e.g., Current Ratio, Debt-to-Equity).

**Why this convention?** Let $W$ be our model's weight vector (a column vector of size $n \times 1$).
If we want to generate predictions ($\hat{Y}$) for *every* company at the exact same time, we compute the matrix multiplication:
$$ \hat{Y} = X \cdot W $$
Dimensions: $(m \times n) \cdot (n \times 1) \rightarrow (m \times 1)$.
The result is exactly an $m \times 1$ column vector containing one prediction for every single company.

### Applying this to the Normalizer
Because $X$ is structured as `[Samples][Features]`, when we calculate the Mean and Standard Deviation for Z-score normalization, we must calculate them **Wise-Column**. 
We want the mean of "Apple's Debt + Ford's Debt..." (Column Mean). We *do not* want the mean of "Apple's Debt + Apple's Profit" (Row Mean), as that incorrectly mixes the units of totally different features.

---

## 2. C++ Project Structure

A standard, scalable C++ project splits code into multiple directories:
*   **`include/` (Headers):** Contains `.h` or `.hpp` files. These files declare *what* a class does (its variables and function signatures) so other files can `#include` and use them without needing to see the full implementation logic.
*   **`src/` (Source):** Contains `.cc` or `.cpp` files. These files hold the actual logic, loops, and memory operations. They implement the promises made in the header files. 

## 3. The Development Environment (VS Code & Clang)

When we initially set up the project, we created a `.vscode/c_cpp_properties.json` file. Here is what is happening behind the scenes so you feel more comfortable coding:

### What is Clang?
`clang++` is the actual **Compiler**. The C++ code you write in VS Code is just text. The processor chip in your computer does not understand English or C++. The compiler (`clang++`) acts as the translator, taking all our `.cc` files, optimizing the mathematics, and stripping them down into 1s and 0s (binary logic) that the machine can execute at massive speeds.

### What is IntelliSense?
IntelliSense is the engine inside your VS Code text editor that provides the smart auto-complete, hovers, and red squiggly error lines when you type code incorrectly. 

### Why did we need `c_cpp_properties.json`?
By default, VS Code doesn't know where your C++ standard libraries are located, or what standard of C++ you are using. By giving it the path to your compiler (`/usr/bin/clang++`) and telling it to use the C++17 standard, we "synced" your editor with your compiler. Now, when you start typing `#include <vector>`, IntelliSense knows exactly where to find that file on your local machine and will proactively show you all the `std::vector` class methods as you code!

---

## 4. Understanding AI Coding Agents & The Antigravity IDE

As your AI pair-programmer, I am not just a chatbot inside your editor—I am an autonomous agent with access to tools. This means you do not have to copy and paste code back and forth or run terminal commands yourself. 

### What I Can Do (My Toolbelt):
1.  **Read & Search:** I can search your entire project for specific functions (`grep`), read your open files, and navigate your directory tree.
2.  **Edit Files:** I can precisely edit lines of code, refactor large blocks, or create entirely new files without you having to touch the keyboard.
3.  **Run Terminal Commands:** I have a background terminal! If you ask me to compile code, run tests, or even install packages (like `sudo apt install xyz`), I can execute those commands, read the hidden terminal output, and fix compilation errors automatically based on what the terminal spat out.
4.  **Create Artifacts:** You've seen me do this! I can create long-term memory files like `project_instructions.md` or this very `learning_notes.md` file so we don't lose track of our macro goals across different chat sessions.

### How to Use Me Effectively (Best Practices):
*   **Give High-Level Goals:** Instead of saying *"Write a for-loop on line 42 of main.cpp that prints x"*, you can say *"I compiled driver.cc but it segfaulted. Can you find the memory leak and fix it?"* I will go read the stack trace, find the file, and edit it.
*   **Ask for Explanations (Like we are doing now!):** You can highlight complicated template metaprogramming code or confusing math and say *"Explain how this affects the AUC score."*
*   **Set Guardrails:** Just like you did by telling me *"Do not generate the code and replace my understanding"*, you can always dictate *how* I interact with you.
*   **Let Me Do the Boring Stuff:** Need 15 boilerplate getter and setter functions? Need a `Makefile` written from scratch? Just ask me to write the boilerplate so you can focus on the core math engine.

### The Standard Workflow:
1. You describe a feature logic or an architectural issue.
2. We iterate on the math/logic in the chat interface.
3. Once agreed, you can either write the code (and I will review it), or ask me to implement it via my file-editing tools.
4. If there is a compilation error, I can see the red squiggly lines via the IDE integration or parse the terminal output, and fix it proactively.

---

## 5. Machine Learning Transformations in Production

### Batch Transform vs. Single Transform
When we wrote the `Normalizer`, we implemented two entirely different `transform` functions via C++ function overloading:
1.  **`transform(const Matrix& x)`**: This is used during **Training**. You feed it massive historical datasets of 10,000 companies. It applies vector math across the entire matrix simultaneously. 
2.  **`transformSingle(const std::vector<double>& x)`**: This is used during **Production (Inference)**. When our Credit Risk application goes live, users will only want to evaluate *one* company at a time. It is terribly inefficient to load a single row of data into a heavy 2D Matrix wrapper. Instead, we just pass an array containing that single company's features, and normalize it instantly using the `means_` and `stds_` we froze into memory during `fit()`.

---

## 6. Financial Domain: Predicting Corporate Default

Before building out a predictive model, a Quant must understand the underlying features. Here are the most prominent financial ratios used in credit risk models (such as the Altman Z-Score) to predict if a company will go bankrupt:

### 1. Return on Assets (ROA)
*   **Formula:** `Net Income / Total Assets`
*   **What it means:** How efficiently is the company using its factories, patents, and cash to generate profit? 
*   **Impact on Default:** High ROA heavily *decreases* probability of default. If they are highly profitable, they can easily service their debt.

### 2. Debt-to-Equity / Total Debt Ratio
*   **Formula:** `Total Liabilities / Total Assets`
*   **What it means:** What percentage of the company is funded by borrowed money (leverage) versus money invested by shareholders?
*   **Impact on Default:** High Debt Ratios severely *increase* probability of default. If a company owes 90 cents for every 1 dollar it owns, a slight dip in revenue will cause them to default on their bond payments.

### 3. Current Ratio (Liquidity)
*   **Formula:** `Current Assets / Current Liabilities`
*   **What it means:** Can the company pay off the debts that are due *within the next 12 months* using cash or assets they can instantly sell?
*   **Impact on Default:** A ratio below 1.0 means the company literally doesn't have the cash to pay its upcoming bills, triggering immediate bankruptcy risks.

### 4. Retained Earnings to Total Assets
*   **Formula:** `Retained Earnings / Total Assets`
*   **What it means:** A measure of cumulative profitability over the company's entire lifespan, acting as a buffer.
*   **Impact on Default:** Young companies (like startups burning venture capital) have very low or negative retained earnings and are statistically much more likely to default than a 100-year-old profitable company like Coca-Cola.

---

## 7. C++ Data Engineering (The DataLoader)

To read raw CSV files, we rely on three powerful C++ concepts:

### What is `std::tuple`?
In C++, a function can normally only `return` one single variable. But what if we scan a CSV file and extract *both* the Feature Matrix ($X$) and the Label Vector ($y$)? We need to return both!
A `std::tuple` is a fixed-size collection of heterogeneous values (values of different types).
```cpp
// Creation
std::tuple<Matrix, std::vector<double>> my_data = {x_matrix, y_vector};

// Extraction (Unpacking)
Matrix X = std::get<0>(my_data);
std::vector<double> y = std::get<1>(my_data);
```

### Reading Files (`std::ifstream` and `std::getline`)
We use an **Input File Stream** (`ifstream`) to open the file, and `getline` to pull out one row of text at a time.
```cpp
#include <fstream>
#include <string>

std::ifstream file("data.csv");
std::string row;

// 'getline' reads characters until it hits a newline (\n)
while (std::getline(file, row)) {
    // 'row' now contains something like: "1001,0.45,0.88,1"
}
```

### Breaking Strings apart (`std::stringstream`)
Once we have a single `row` of text, we treat that string like a mini-file using `stringstream`. We then use `getline` again, but this time we tell it to stop exactly when it hits a comma (`,`).
```cpp
#include <sstream>

std::stringstream ss(row);
std::string token;

// Extract characters until hitting a comma
while (std::getline(ss, token, ',')) {
    // token is now "0.45" (still a string!)
    
    // Convert the string to a math double:
    double value = std::stod(token);
}
```

### 4. Handling Missing Data (`NaN`)
In the real world, datasets are never perfect. Sometimes a company hasn't reported their quarterly earnings yet, or the database had an error. When this data is exported to a CSV, the missing cells are often filled with the text `"NaN"` (Not a Number), `"NA"`, or left completely blank `,,`.

If you feed `"NaN"` into a Python string parser, it handles it gracefully. But in C++, `std::stod("NaN")` evaluates to a mathematical float type representing `NaN`. 
If a single `NaN` makes its way into your Matrix, any math you perform on it (like Z-score normalization or gradient descent) will result in more `NaN`s, completely destroying your model's weights.
**The Solution:** You must explicitly check your string tokens in C++ using `if (token == "NaN" || token == "nan")` and either discard that row entirely or replace it with the column average (Imputation).

---

## 8. Model Generalization: L2 Regularization (Ridge)

When building a Quant model, you are not trying to memorize the past; you are trying to predict the future.

### The Problem of Overfitting
Imagine predicting bankruptcy using two features: 
1. **Debt Ratio** (Actually useful)
2. **CEO's Zodiac Sign** (Completely useless noise)

By sheer random chance in your training dataset, every single company whose CEO is a Scorpio happens to have gone bankrupt. 
If you use **Standard Logistic Regression**, the model is an absolute perfectionist. It wants its training loss to be exactly zero. It will look at the Scorpio data, assume it’s a perfect predictor, and assign a massive, dominating weight (e.g., $w_{\text{zodiac}} = 9000.0$).
When deployed to the real world, it sees a healthy company like Apple, notices the CEO is a Scorpio, and immediately predicts doom because that $9000.0$ weight overpowers the actual financial math. This is **Overfitting**.

### The Solution: Taxing the Weights (L2 Penalty)
L2 Regularization changes the model's objective. We add a penalty term to the Loss Function:
$$ J(w) = \text{LogLoss}(w) + \frac{\lambda}{2} \sum w_j^2 $$

The model must now balance returning accurate predictions while keeping all weights as close to $w=0$ as possible. The hyperparameter **$\lambda$ (Lambda)** controls how strict the penalty is.

If the model tries to set $w = 9000.0$, the penalty squares that number ($81,000,000$) and blows up the Loss. The model is forced to spread its "trust" across multiple features, giving them all small, reasonable weights. This acts as a skeptical boss, forcing the model to look at the broader picture and generalize better to unseen data.

### The C++ Calculus Update
To implement this in Gradient Descent, we just take the derivative of the penalty term:
$$ \frac{d}{dw_j} \left( \frac{\lambda}{2} w_j^2 \right) = \lambda w_j $$

In our C++ training loop (`logistic_regression.cc`), instead of updating weights solely via the loss gradient:
```cpp
weights[j] -= learning_rate * log_loss_gradient;
```
We simply add the L2 derivative to shrink the weight slightly every step:
```cpp
weights[j] -= learning_rate * (log_loss_gradient + lambda_ * weights[j]);
```

---

## 9. Gradient Descent: Batch vs. Stochastic (SGD)

When training a machine learning model, the "Gradient" is the direction we need to move our weights to lower the Loss. But *how often* should we update our weights? This choice vastly changes the speed, accuracy, and memory usage of our C++ engine.

### 1. Batch Gradient Descent (What We Currently Use)
In Batch Gradient Descent, the model evaluates the error for **all 1,000 companies in the dataset** before taking a single step.
*   **The Math:** It calculates the gradient for Row 1, Row 2... Row 1000, adds them all together into a `grad_w` accumulator, and then averages them out to update the weights exactly once per epoch.
*   **Pros:** The mathematical path to the absolute lowest Loss (global minimum) is perfectly smooth and stable. It takes a perfect step every time. 
*   **Cons:** It is painfully slow for massive datasets. If you have 10 million rows, calculating the 10-million-row average just to take *one tiny step* takes immense computational time and requires all data to fit into RAM simultaneously.

### 2. Stochastic Gradient Descent (SGD)
In Stochastic Gradient Descent (SGD), the model updates its weights **immediately after looking at a single row**.
*   **The Math:** It looks at Row 1 (Apple), calculates the error, and instantly steps its weights. Then it looks at Row 2 (Ford), calculates the error, and instantly steps again. Therefore, in 1 epoch over a 1000-row dataset, the weights are updated 1000 separate times!
*   **Pros:** It is blisteringly fast. Instead of getting stuck calculating a massive average, it sprints toward the solution. It also allows models to train on infinite streams of live data, because you don't need to hold the entire dataset in RAM at once.
*   **Cons:** Because the model steps based on *one single company's* data, its path toward the minimum loss is incredibly erratic and noisy. If it looks at an outlier company, it will jump in entirely the wrong direction before correcting itself on the next row. It rarely settles perfectly at the absolute minimum loss.

### 3. Mini-Batch Gradient Descent (The Industry Standard)
Modern Neural Networks and Deep Learning rarely use pure Batch or pure SGD. They use a compromise called Mini-Batch.
*   **The Math:** We divide our 10,000-row dataset into small "batches" of, say, 32 or 64 rows. The model looks at 32 rows, averages the gradient, and updates the weights.
*   **Why it wins:** It combines the lightning speed of updating weights frequently (SGD) with the mathematical stability of taking an average (Batch). Furthermore, splitting data into small chunks of 32 or 64 perfectly utilizes the parallel processing architecture of modern GPUs!

---

## 10. Architecting a C++ ML Project from Scratch

How do you know where to put functions like `train_test_split()`? Should it go in `train.cc`, `Matrix`, or `DataLoader`?

When building professional software, engineers follow the **Single Responsibility Principle (SRP)** from the SOLID design guidelines. 
Every class should have exactly one job, and one reason to change.

### 1. The Core Data Structure (`Matrix`)
*   **Its Job:** Linear Algebra. Pure mathematical operations. 
*   **Why we don't put `split()` here:** A mathematical matrix shouldn't care about "Training" or "Testing." It just holds numbers and multiplies them. If you add machine-learning specific concepts to `Matrix`, you ruin its ability to be reused in another project (like building a 3D Graphics engine).

### 2. The Algorithm Engine (`LogisticRegression` / `Normalizer`)
*   **Its Job:** Statistics, gradients, loss functions.
*   **Why we don't put `split()` here:** The engine expects the data to already be perfectly prepared before `train()` is ever called.

### 3. The Utility Belt (`DataLoader`)
*   **Its Job:** Everything related to taking messy external data and transforming it into clean input for the engine.
*   **Why `split()` belongs here:** Slicing a dataset into 80% and 20% chunks is a fundamental data-preparation step. If a user wants to load data, they almost always want to split it. This is exactly where it belongs!

### 4. The Orchestrator (`train.cc` / `driver.cc`)
*   **Its Job:** The boss. It instantiates all the classes above and plugs them together like LEGO bricks.
*   **Why we don't put `split()` here:** We want `train.cc` to be as clean and short as possible. If we put a massive 20-line `for` loop in `train.cc` just to split data, it becomes impossible to read the high-level architecture of what the script is actually doing.

