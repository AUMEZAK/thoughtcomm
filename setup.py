from setuptools import setup, find_packages

setup(
    name="thoughtcomm",
    version="0.1.0",
    description="Reproduction of 'Thought Communication in Multiagent Collaboration' (NeurIPS 2025)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.27.0",
        "scipy>=1.11.0",
        "sympy>=1.12",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
    ],
)
