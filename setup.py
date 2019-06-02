from setuptools import setup, find_packages
setup(name='wookie',
      version='0.35',
      description='MAAAAAAAAAAH',
      url='http://github.com/ogierpaul/wookie',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'scikit-learn',
          'pandas',
          'numpy',
          'fuzzywuzzy',
          'pytest',
          'dask',
          'elasticsearch'
      ],
      zip_safe=False)
