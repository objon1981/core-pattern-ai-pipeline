# Packaging/install script (optional)

from setuptools import setup, find_packages

setup(
    name="core_pattern_ai_pipeline",
    version="0.1.0",
    author="John King",
    author_email="sogumint@gmail.com",
    description="An AI pipeline inspired by natural processes using wavelets, graphs, GATs, and Fibonacci logic with LLM explanations and user feedback.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/core-pattern-ai-pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "networkx",
        "torch",
        "transformers",
        "tqdm",
        "pandas",
        "flask",  # For dashboard (if web-based)
        "sqlalchemy",  # For feedback_db
    ],
    extras_require={
        "dev": [
            "pytest",
            "jupyter",
            "black",
            "flake8",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.8',
)
