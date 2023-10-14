from setuptools import setup, find_packages

setup(
    name="chinese_checkers",
    version="0.1",
    author="Dakota James Parker",
    description="A Chinese Checkers game implementation.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    extras_require={
        "test": ["pytest", "pytest-cov"],
    },
)