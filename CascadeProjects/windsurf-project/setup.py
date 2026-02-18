from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mat-processor',
    version='0.1.0',
    description='A Python package for working with .mat files',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
        'h5py>=3.1.0',
        'seaborn>=0.11.0',
    ],
    entry_points={
        'console_scripts': [
            'mat-processor=mat_processor.cli:main',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
    ],
    keywords='matlab mat data-analysis visualization',
    project_urls={
        'Source': 'https://github.com/yourusername/mat-processor',
        'Bug Reports': 'https://github.com/yourusername/mat-processor/issues',
    },
)
