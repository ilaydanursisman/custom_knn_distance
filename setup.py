from setuptools import setup, find_packages

setup(
    name='custom_knn_distance',
    version='0.1.0',
    author='İlayda Nur Şişman',
    author_email='ilaydasisman65@gmail.com',
    description='Custom distance metric for KNN classification based on local density adaptation',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ilaydanursisman/custom_knn_distance', 
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)