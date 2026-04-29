from setuptools import setup, find_packages

setup(
    name='aug_accuracy',
    packages=find_packages(),
    version='0.1',
    description='Set of tools for reproducible training and testing of claffication networks.',
    author='Vladyslav Andriiashen',
    author_email='vladyslav.andriiashen@cwi.nl',
    url='https://github.com/vandriiashen/aug_accuracy',
    package_dir={'aug_accuracy': 'aug_accuracy'},
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'imageio',  # imageio 2.15 has a bug with resolution tags
        'scikit-learn',
        'tensorboard'
    ]
)
