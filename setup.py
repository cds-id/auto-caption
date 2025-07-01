from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="auto-caption",
    version="0.1.0",
    author="Auto Caption Team",
    author_email="info@ciptadusa.com",
    description="AI-powered emotion-aware caption generation for short-form videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cds-id/auto-caption",
    project_urls={
        "Bug Tracker": "https://github.com/cds-id/auto-caption/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "auto-caption=auto_caption.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
