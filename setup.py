from setuptools import setup, find_packages

setup(
    name='optgan',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Adicione suas dependências aqui
    ],
    entry_points={
        'console_scripts': [
            'run = optgan.main:main',
        ],
    },
)