# Fashion MNIST GAN

A simple Generative Adversarial Network (GAN) implemented in PyTorch to generate Fashion MNIST images.

## Project Structure

- `src/` – Source code for models and utility functions.
- `notebooks/` – Jupyter notebook for experiments and visualizations.
- `data/` – (empty, add data here if needed)
- `assets/` – Generated sample images and plots.

## Setup

1. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script to see the GAN’s generator output:

```bash
python src/gan.py
```

Or open the notebook for exploration:

```bash
jupyter notebook notebooks/FashionMNIST_GAN.ipynb
```

## Results

Sample generated images:

![Sample](assets/example.png)

## License

MIT
