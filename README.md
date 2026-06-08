# InfiniBench HuggingFace Lite

InfiniBench HuggingFace Lite is a small, beginner-friendly project for evaluating visual spatial reasoning with the pre-generated HuggingFace dataset [`Haoming645/infinibench`](https://huggingface.co/datasets/Haoming645/infinibench).

This Lite version only streams existing image samples, saves a small local subset, creates CSV templates for questions and answers, calculates accuracy, and plots simple results.

It does **not** use Blender, FFmpeg, GPU/CUDA, the full InfiniBench 3D scene generation pipeline, or local VLM model loading.

## What You Can Do

1. Stream image samples from HuggingFace without downloading the full dataset.
2. Save a small number of images locally, such as 50 or 100.
3. Create a CSV file with image paths and labels.
4. Create a simple visual spatial reasoning question template.
5. Manually fill ground-truth answers.
6. Add manual or API-collected model answers later.
7. Mark each answer as correct or wrong.
8. Calculate overall and task-wise accuracy.
9. Generate a simple bar chart of accuracy by task type.

## What This Version Does Not Include

The original InfiniBench project includes an advanced 3D scene generation and rendering workflow. That full pipeline can involve Blender, FFmpeg, procedural scene generation, rendering, camera trajectories, and advanced dependencies.

This repository has been simplified. The original advanced files are archived in `archive_original/` so they do not distract from the lightweight HuggingFace workflow.

You do not need `archive_original/` to run this Lite project.

## Requirements

You only need Python and the small packages listed in `requirements.txt`:

```text
datasets
pillow
pandas
matplotlib
tqdm
```

No GPU is required. A normal laptop CPU or Google Colab CPU runtime is enough.

## Installation

From the project root:

```bash
python -m venv .venv
```

Activate the environment.

On Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Folder Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 01_load_samples.py
│   ├── 02_create_question_template.py
│   ├── 03_evaluate_results.py
│   └── 04_plot_results.py
├── notebooks/
│   └── infinibench_hf_lite_demo.ipynb
├── data/
│   ├── samples/
│   ├── infinibench_samples.csv
│   ├── infinibench_questions_template.csv
│   └── results_template.csv
├── outputs/
│   ├── accuracy_by_task.csv
│   └── accuracy_by_task.png
└── archive_original/
    └── original advanced InfiniBench files
```

`outputs/accuracy_by_task.csv` and `outputs/accuracy_by_task.png` are created after running evaluation and plotting.

## Step 1: Load HuggingFace Samples

This streams the dataset and saves only the number of images you request.

```bash
python scripts/01_load_samples.py --num-samples 100
```

Useful options:

```bash
python scripts/01_load_samples.py \
  --num-samples 50 \
  --output-dir data/samples \
  --metadata-csv data/infinibench_samples.csv
```

Output:

```text
data/samples/
data/infinibench_samples.csv
```

The metadata CSV has these columns:

```text
id,image_path,label
```

## Step 2: Create Question Template

```bash
python scripts/02_create_question_template.py
```

This reads:

```text
data/infinibench_samples.csv
```

It creates:

```text
data/infinibench_questions_template.csv
data/results_template.csv
```

Each image gets simple default questions:

```text
How many chairs are visible in the image?
Is there a table visible in the image?
Is the scene cluttered?
Are any objects partially occluded?
```

Task types:

```text
counting
object_presence
clutter_judgment
occlusion_judgment
```

The CSV columns are:

```text
id,image_path,label,task_type,question,ground_truth,model_answer,correct
```

## Step 3: Manually Evaluate VLM Answers

Open `data/results_template.csv` in Excel, Google Sheets, LibreOffice, or a text editor.

Fill these columns:

```text
ground_truth
model_answer
correct
```

Use `correct = 1` when the model answer is correct.

Use `correct = 0` when the model answer is wrong.

Example:

```text
ground_truth,model_answer,correct
yes,yes,1
3,2,0
no,no,1
```

You can collect `model_answer` manually or from any external VLM API. This project does not call any VLM API by default.

## Step 4: Calculate Accuracy

After filling the `correct` column:

```bash
python scripts/03_evaluate_results.py --results-csv data/results_template.csv
```

Output:

```text
outputs/accuracy_by_task.csv
```

The script prints:

```text
overall accuracy
task-wise accuracy
number of questions per task type
```

Blank `correct` values are ignored during accuracy calculation.

## Step 5: Plot Results

```bash
python scripts/04_plot_results.py
```

Output:

```text
outputs/accuracy_by_task.png
```

This creates a simple matplotlib bar chart. It does not use seaborn or complicated styling.

## Example Workflow

```bash
pip install -r requirements.txt
python scripts/01_load_samples.py --num-samples 50
python scripts/02_create_question_template.py
```

Then manually edit:

```text
data/results_template.csv
```

Fill:

1. `ground_truth`
2. `model_answer`
3. `correct` as `1` or `0`

Then run:

```bash
python scripts/03_evaluate_results.py --results-csv data/results_template.csv
python scripts/04_plot_results.py
```

Final outputs:

```text
outputs/accuracy_by_task.csv
outputs/accuracy_by_task.png
```

## Google Colab Instructions

Use a CPU runtime. GPU is not needed.

In Colab:

```bash
!git clone <your-repo-url>
%cd <your-repo-folder>
!pip install -r requirements.txt
```

Then run:

```bash
!python scripts/01_load_samples.py --num-samples 50
!python scripts/02_create_question_template.py
```

Open or download `data/results_template.csv`, fill the answer columns, upload it back if needed, then run:

```bash
!python scripts/03_evaluate_results.py --results-csv data/results_template.csv
!python scripts/04_plot_results.py
```

You can also open:

```text
notebooks/infinibench_hf_lite_demo.ipynb
```

## Troubleshooting

If `datasets` is missing:

```bash
pip install -r requirements.txt
```

If the dataset does not load, check your internet connection. Streaming still needs internet access because images are read from HuggingFace as you iterate.

If `data/infinibench_samples.csv` is empty, rerun:

```bash
python scripts/01_load_samples.py --num-samples 50
```

If evaluation says no usable `correct` values were found, make sure the `correct` column contains `1` or `0`.

If the plot script cannot find `outputs/accuracy_by_task.csv`, run evaluation first:

```bash
python scripts/03_evaluate_results.py --results-csv data/results_template.csv
```

## Notes

This project intentionally keeps the workflow simple for students. It uses the pre-generated HuggingFace dataset instead of generating new 3D scenes.

You only download packages when you run `pip install -r requirements.txt`, and you only stream dataset samples when you run `scripts/01_load_samples.py`.
