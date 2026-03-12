# Real-World Software Projects: The Polyglot Pipeline

When learning to code in school or tutorials, you usually write everything in one single file, using one language (like `main.cpp` or `script.py`). But in the real world of Quantitative Finance, Machine Learning, and Big Tech, projects are almost never written in just one language. 

They are **Polyglot** (multi-language) pipelines. 

This document answers common questions a junior programmer might have about how these projects are actually structured, executed, and why we mix languages.

---

## 1. How do Python and C++ fit together?
You know that C++ code needs to be "compiled" into a binary executable (like our `credit_risk_app`) using a compiler (`g++` or `clang++`) before it can run. Python, on the other hand, is an "interpreted" language—it runs directly from the script file line-by-line using the `python3` command.

**How do they communicate?**
They usually don't talk to each other directly in the same computer memory space. Instead, they hand files back and forth like a factory assembly line!

**The Factory Pipeline:**
1.  **The Scout (Python):** Python is slow but has amazing libraries. You write a script (`generate_data.py` or `fetch_stock_prices.py`) that reaches out to the internet, downloads financial data, cleans it up, and saves it as a standard text file (`data/credit_risk_dataset.csv`).
2.  **The Engine (C++):** C++ is extremely fast but harder to write. You compile your C++ code into a binary. That fast binary is programmed to read the `csv` file, perform a billion mathematical matrix operations in a split second, and then spit out the final answers into a new file, like `report/results.txt`.
3.  **The Presenter (Python or Web):** You might have another Python script or a web dashboard that reads `results.txt` and draws beautiful graphs for your investors.

---

## 2. Who presses the "Run" button on all these different things?
If you have a Python script to get data, a C++ command to compile, and a C++ command to run the model... do you type all of them manually in the terminal every time? **No!**

In a real project, engineers use "Build Systems" and "Orchestration Tools" to automate the factory.

### The `Makefile` (The C++ Builder)
We just created a `Makefile`. This is a classic C++ tool. Instead of typing:
```bash
g++ -std=c++17 src/main.cc src/math.cc src/loader.cc -o bin/app
```
You just type `make` in your terminal. The `Makefile` acts as a robotic blueprint that tells the computer exactly how to link and compile the C++ files automatically.

### Shell Scripts (`.sh` files) (The Orchestrator)
To tie the Python and C++ pieces completely together, engineers write "Shell Scripts" (usually named something like `run_pipeline.sh`). This is just a list of terminal commands saved in a simple file.

A real `run_pipeline.sh` might look exactly like this:
```bash
#!/bin/bash
echo "Starting Pipeline..."

# 1. Run Python to get the data
python3 scripts/generate_data.py

# 2. Compile the C++ Engine (using the Makefile)
make

# 3. Run the C++ Engine and save the output
./bin/credit_risk_app > report/output.txt

echo "Pipeline Finished!"
```
Now, the engineer just types `./run_pipeline.sh` one single time. The computer spins up the *entire* factory, running Python, compiling C++, and executing the binary automatically.

---

## 3. Why not just write the whole thing in C++ or Python?
*   **Why not all Python?** If a hedge fund wrote their High-Frequency Trading engine in Python, it would execute trades 100x slower than their competitors using C++. They would lose millions of dollars.
*   **Why not all C++?** If that same hedge fund tried to write the script to download Stock Data from a web server API in C++, it would take a senior engineer 3 weeks to write memory-safe code. In Python, it takes 3 lines of code and 5 minutes.

**In software engineering, we use the best tool for the specific job.** 
Python is the best tool for data fetching, rapid prototyping, and scripting. C++ is the best tool for heavy, low-latency, core mathematics.

---

## 4. How does the Directory Structure work?
Because projects use multiple languages, keeping things organized is critical so tools don't step on each other's toes. This is why our project is split up:
*   `src/` and `include/` are strictly for C++ source code.
*   `scripts/` is for helper Python or Bash scripts.
*   `data/` holds the CSV files that act as the physical "bridge" between Python and C++.
*   `bin/` holds the compiled binary C++ executables. You never push this folder to GitHub (which is why we put it in `.gitignore`), because compiling must happen uniquely on whatever computer physically runs the code.

By enforcing this strict separation, a Data Scientist can safely edit the Python scripts in `scripts/` without accidentally breaking the core C++ Math Engine in `src/`.

---

## 5. Why Does C++ still have to parse the CSV? (The I/O Boundary)
A very common question: *"If Python is fetching the data, why can't Python just feed the numbers directly into the C++ variables? Why string-parse a CSV in C++?"*

Because **Memory Spaces are Isolated**. When you run a Python script, the Operating System gives it a closed sandbox of RAM. When you run a compiled C++ binary, it gets a completely separate sandbox. Python cannot physically reach into C++'s memory to hand it an integer. 

The standard bridge between two different languages is **File I/O (Input/Output)** or **Network Sockets**. 
1. Python writes the data to the hard drive in a universal format (like CSV or JSON).
2. The C++ binary opens that file off the hard drive, parses the universal text, and loads it into its own high-speed memory structures (like our `Matrix` class).

## 6. Industry Jargon Cheat Sheet
When interviewing or working at a firm, people will rarely say "Python script" or "C++ engine". They use specific architectural terms:

*   **ETL (Extract, Transform, Load):** What our Python script does. It **E**xtracts data from the internet, **T**ransforms it (cleans up bad data), and **L**oads it into a database or CSV.
*   **Data Lake / Data Warehouse:** The massive hard drives where Python dumps all the raw CSVs.
*   **Inference:** Using the trained C++ model to predict new, live data. Training happens once; Inference happens millions of times a day.
*   **Latency:** The time it takes between receiving a piece of data and outputting a prediction. Hedge funds use C++ to achieve "Ultra-Low Latency" (microseconds).
*   **Polyglot:** A repository that contains multiple programming languages.
*   **Batch Processing:** Our `train.cc` is a Batch script. It reads 1000 rows at a time.
*   **Streaming / Event-Driven:** What a live trading system uses. It reads data row-by-row instantly as it happens on the stock exchange.
