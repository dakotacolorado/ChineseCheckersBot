from setuptools import setup, find_packages

setup(
    name="chinese_checkers",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    extras_require={
        "test": ["pytest", "pytest-cov"],  # add your testing dependencies here
    },
)