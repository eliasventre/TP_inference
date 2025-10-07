# TP_inference

This repository supports the second part of the **CompSysBio 2025** hands-on session:
**Simulation and reverse-engineering of a mechanistic model of gene expression.**

---

## 🧰 Installation

You can install the project either using **Conda** (recommended) or **pip**.

---

### 🔹 Option 1 — Using Conda (recommended)

#### 1️⃣ Clone or download the project

```bash
git clone https://github.com/eliasventre/TP_inference.git
cd TP_inference
```

#### 2️⃣ Create and activate the environment

```bash
conda env create -f environment.yml
conda activate TP_inference
```

> 💡 *Note:* You don’t need to run `conda create` manually —
> the `environment.yml` file already defines the environment name and dependencies.

---

### 🔹 Option 2 — Using pip

#### 1️⃣ Clone or download the project

```bash
git clone https://github.com/eliasventre/TP_inference.git
cd TP_inference
```

#### 2️⃣ (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# .\venv\Scripts\activate   # Windows
```

#### 3️⃣ Install the dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Once the environment is ready, you can run the main script or notebook for the session.

Example:

```bash
python simlulate_distributions.py
```

```bash
python inference.py -m correlations
```

```bash
python evaluation.py
```

