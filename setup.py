from setuptools import setup, find_packages
import smoove

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
                'numpy',
                'scipy',
                'numba',
                'pytest >= 6.2.2',
                'autograd',
            ]


setup(
     name='smoove',
     version=smoove.__version__,
     author="Landman Bester",
     author_email="lbester@sarao.ac.za",
     description="Miscelaneous fast smoothing utilities",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/landmanbester/smoove",
     packages=find_packages(),
     python_requires='>=3.7',
     install_requires=requirements,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
